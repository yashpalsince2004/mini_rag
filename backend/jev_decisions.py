"""
backend/jev_decisions.py - TypeSafe Jev Decision Layer via OpenRouter Decisions API.

Responsibilities:
1. Decision 1 (Query Routing): Classifies user query into typed categories (document_question, greeting, help, unsupported, clarification_needed).
2. Decision 2 (Retrieval Gating): Decides whether ChromaDB vector search should execute.
3. Decision 3 (Context Quality): Evaluates if retrieved chunks contain sufficient factual evidence.
4. Decision 4 (Generation Gating): Authorizes or blocks LLM (Gemini) answer synthesis.
5. Error Handling & Conservative Fallback: Safely degrades to deterministic local heuristics when OpenRouter is offline or unconfigured, preventing hallucinations.
"""

import os
import re
from dataclasses import dataclass, asdict
from typing import Optional, Any
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import httpx

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# OpenRouter Decisions API configuration
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_JEV_MODEL = os.getenv("JEV_MODEL", "~typesafe/jev-latest")


# =====================================================================
# Decision Data Structures
# =====================================================================

@dataclass
class RoutingDecision:
    query_type: str                   # 'document_question' | 'greeting' | 'help' | 'unsupported' | 'clarification_needed'
    should_retrieve: bool
    confidence: float
    probabilities: dict[str, float]
    is_fallback: bool = False
    engine: str = "openrouter-jev"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContextSufficiencyDecision:
    context_quality: str              # 'sufficient' | 'insufficient' | 'uncertain'
    should_generate: bool
    confidence: float
    probabilities: dict[str, float]
    noul_sufficiency: Optional[float] = None
    is_fallback: bool = False
    engine: str = "openrouter-jev"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# Reusable OpenRouter Jev Client
# =====================================================================

class JevClient:
    """
    Lightweight client for interacting with TypeSafe Jev System One decision models
    via OpenRouter's specialized Decisions API (POST /api/alpha/decisions).
    """
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        base_url: str = OPENROUTER_DECISIONS_URL,
        timeout: float = 15.0
    ):
        self.api_key = api_key.strip()
        self.model = model or os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Mini PDF RAG Chatbot"
        }

    def decide(self, state: Any, questions: dict) -> dict:
        """
        Sends a typed decision request to OpenRouter's Decisions endpoint.
        Returns the parsed 'answers' dictionary from the response.
        Raises RuntimeError on API failure.
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, json=payload, headers=self.headers)
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter Decisions API error ({response.status_code}): {response.text}"
                )

            data = response.json()
            answers = data.get("answers")
            if not answers:
                raise RuntimeError(f"OpenRouter response missing 'answers' field: {data}")

            return answers

    def classify_query(self, query: str, filename: Optional[str] = None) -> RoutingDecision:
        """
        Sends Decision 1 (intent classification) and Decision 2 (retrieval gating) to Jev.
        """
        state = {
            "user_query": query,
            "active_document": filename or "Selected document"
        }

        questions = {
            "query_type": {
                "type": "choice",
                "instructions": "Classify the user query intent for a document question-answering assistant.",
                "criteria": {
                    "document_question": "Questions asking about information, facts, concepts, or details that can be answered from the document or PDF.",
                    "greeting": "Casual greetings, hellos, good mornings, or introductory pleasantries.",
                    "help": "Asking how to use this tool, what it does, or asking for instructions on using the assistant.",
                    "unsupported": "Out-of-scope requests such as writing code, generating creative stories, checking real-time weather, or general knowledge unrelated to documents.",
                    "clarification_needed": "Empty, garbled, or completely ambiguous input that cannot be interpreted without clarification."
                }
            },
            "should_retrieve": {
                "type": "noul",
                "instructions": "Should the assistant perform semantic retrieval from the selected document to answer this query?",
                "criteria": {
                    "true": "The user is asking a factual question about the document content, topics, or subject matter (e.g. book details, tests, rules, sections, scores).",
                    "false": "The query is a greeting, polite phrase, help request, coding instruction, or unrelated request that does not need document retrieval."
                }
            }
        }

        answers = self.decide(state=state, questions=questions)
        choice_ans = answers.get("query_type", {})
        noul_ans = answers.get("should_retrieve", {})

        # Extract choice results
        selected_choice = choice_ans.get("choice", "document_question")
        conf = float(choice_ans.get("confidence", 0.90))
        probs = {k: float(v) for k, v in choice_ans.get("probabilities", {}).items()}

        # Extract noul probability (handles both 'probability' and 'noul' keys)
        noul_val = noul_ans.get("probability", noul_ans.get("noul", 0.5))
        noul_prob = float(noul_val) if noul_val is not None else 0.5

        # Retrieval gating rule
        should_retrieve = (selected_choice == "document_question") and (noul_prob >= 0.35)

        return RoutingDecision(
            query_type=selected_choice,
            should_retrieve=should_retrieve,
            confidence=conf,
            probabilities=probs,
            is_fallback=False,
            engine=f"openrouter-jev ({self.model})",
            reason=f"Jev classified query as {selected_choice} (retrieve noul: {noul_prob:.2f})."
        )

    def evaluate_context(
        self,
        query: str,
        retrieved_chunks: list[dict],
        filename: Optional[str] = None
    ) -> ContextSufficiencyDecision:
        """
        Sends Decision 3 (context quality) and Decision 4 (generation gating) to Jev.
        """
        state = {
            "question": query,
            "document": filename or "Selected document",
            "retrieved_chunks": [
                {
                    "chunk_index": i + 1,
                    "page": chunk.get("page", 0),
                    "distance": round(chunk["distance"], 4) if chunk.get("distance") is not None else None,
                    "text": chunk.get("text", "")
                }
                for i, chunk in enumerate(retrieved_chunks)
            ]
        }

        questions = {
            "context_quality": {
                "type": "choice",
                "instructions": "Based on the retrieved document chunks, does the context contain sufficient factual information to answer the user's question accurately?",
                "criteria": {
                    "sufficient": "The retrieved chunks explicitly state facts, details, or explanations that directly address the question.",
                    "insufficient": "The retrieved chunks do not contain enough information to answer the question, or are only tangentially related.",
                    "uncertain": "The chunks partially touch upon the topic but are missing crucial facts or leave the answer ambiguous."
                }
            },
            "should_generate": {
                "type": "noul",
                "instructions": "Should the answering model be allowed to generate a factual answer based on these retrieved chunks?",
                "criteria": {
                    "true": "The retrieved chunks contain verifiable evidence to answer the question without hallucination.",
                    "false": "The retrieved chunks lack necessary facts; answering would require guessing or hallucinating."
                }
            }
        }

        answers = self.decide(state=state, questions=questions)
        choice_ans = answers.get("context_quality", {})
        noul_ans = answers.get("should_generate", {})

        selected_quality = choice_ans.get("choice", "insufficient")
        conf = float(choice_ans.get("confidence", 0.85))
        probs = {k: float(v) for k, v in choice_ans.get("probabilities", {}).items()}

        noul_val = noul_ans.get("probability", noul_ans.get("noul", 0.5))
        noul_prob = float(noul_val) if noul_val is not None else 0.5

        # Generation control rule
        should_generate = (selected_quality == "sufficient") or (selected_quality == "uncertain" and noul_prob >= 0.65)

        return ContextSufficiencyDecision(
            context_quality=selected_quality,
            should_generate=should_generate,
            confidence=conf,
            probabilities=probs,
            noul_sufficiency=noul_prob,
            is_fallback=False,
            engine=f"openrouter-jev ({self.model})",
            reason=f"Context evaluated as {selected_quality} (noul probability: {noul_prob:.2f})."
        )


# =====================================================================
# Client Factory
# =====================================================================

def get_jev_client() -> Optional[JevClient]:
    """
    Returns an initialized JevClient if OPENROUTER_API_KEY is configured.
    Returns None if missing or placeholder.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key.strip().startswith("your_"):
        return None

    try:
        model = os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL)
        return JevClient(api_key=api_key.strip(), model=model)
    except Exception as e:
        print(f"[JEV] Warning: Failed to initialize JevClient ({e}). Falling back.")
        return None


# =====================================================================
# Public Pipeline Decision Functions
# =====================================================================

def classify_query_intent(query: str, filename: Optional[str] = None) -> RoutingDecision:
    """
    Decision 1: Evaluates user question with Jev to determine request type.
    Decision 2: Determines whether retrieval should run.
    """
    trimmed = query.strip()
    if not trimmed:
        return RoutingDecision(
            query_type="clarification_needed",
            should_retrieve=False,
            confidence=1.0,
            probabilities={"clarification_needed": 1.0},
            is_fallback=True,
            engine="validator",
            reason="Query is empty."
        )

    client = get_jev_client()
    if client:
        try:
            print(f"[JEV] Invoking OpenRouter Jev model ({client.model}) for query classification...")
            decision = client.classify_query(trimmed, filename=filename)
            print(f"[JEV] Query route: {decision.query_type}")
            print(f"[JEV] Retrieval required: {'true' if decision.should_retrieve else 'false'}")
            return decision
        except Exception as e:
            print(f"[JEV] Warning: OpenRouter API call failed ({e}). Using conservative fallback.")

    # Graceful deterministic fallback when OPENROUTER_API_KEY is unset or API fails
    decision = _fallback_query_routing(trimmed)
    print(f"[JEV] Query route: {decision.query_type}")
    print(f"[JEV] Retrieval required: {'true' if decision.should_retrieve else 'false'}")
    return decision


def evaluate_context_sufficiency(
    query: str,
    retrieved_chunks: list[dict],
    filename: Optional[str] = None
) -> ContextSufficiencyDecision:
    """
    Decision 3: Assesses retrieval quality over the top-K chunks.
    Decision 4: Decides whether LLM generation should proceed.
    """
    if not retrieved_chunks:
        print("[JEV] Context sufficient: false (no chunks retrieved)")
        return ContextSufficiencyDecision(
            context_quality="insufficient",
            should_generate=False,
            confidence=1.0,
            probabilities={"insufficient": 1.0, "sufficient": 0.0, "uncertain": 0.0},
            noul_sufficiency=0.0,
            is_fallback=True,
            engine="validator",
            reason="No chunks retrieved from ChromaDB."
        )

    client = get_jev_client()
    if client:
        try:
            print(f"[JEV] Invoking OpenRouter Jev model ({client.model}) for context sufficiency assessment...")
            decision = client.evaluate_context(query, retrieved_chunks, filename=filename)
            is_suff = (decision.context_quality == "sufficient")
            print(f"[JEV] Context sufficient: {'true' if is_suff else 'false'}")
            return decision
        except Exception as e:
            print(f"[JEV] Warning: OpenRouter API call failed ({e}). Using conservative fallback.")

    # Graceful deterministic fallback when OPENROUTER_API_KEY is unset or API fails
    decision = _fallback_context_sufficiency(query, retrieved_chunks)
    is_suff = (decision.context_quality == "sufficient")
    print(f"[JEV] Context sufficient: {'true' if is_suff else 'false'}")
    return decision


# =====================================================================
# Conservative Local Heuristic Fallbacks (Zero Hallucination Guarantee)
# =====================================================================

def _fallback_query_routing(query: str) -> RoutingDecision:
    """
    Deterministic fallback that mimics Jev's routing criteria when OpenRouter is unconfigured.
    """
    lower = query.lower().strip()
    clean = re.sub(r'[^\w\s]', '', lower)

    # Greeting check
    greeting_tokens = {"hi", "hello", "hey", "hiya", "howdy", "good morning", "good evening", "good afternoon"}
    if clean in greeting_tokens or clean.startswith("hello ") or clean.startswith("hi "):
        return RoutingDecision(
            query_type="greeting",
            should_retrieve=False,
            confidence=0.95,
            probabilities={"greeting": 0.95, "document_question": 0.05},
            is_fallback=True,
            engine="fallback-heuristic",
            reason="Recognized standard greeting pattern."
        )

    # Help check
    if clean in {"help", "how does this work", "what can you do", "instructions"} or "how do i use" in clean:
        return RoutingDecision(
            query_type="help",
            should_retrieve=False,
            confidence=0.90,
            probabilities={"help": 0.90, "document_question": 0.10},
            is_fallback=True,
            engine="fallback-heuristic",
            reason="Recognized help request pattern."
        )

    # Unsupported check (e.g. write code, weather, tell joke)
    unsupported_patterns = [
        r"\b(write|create|code|program)\b.*\b(game|app|script|function|code)\b",
        r"\b(weather|temperature|forecast)\b",
        r"\b(tell|make)\b.*\b(joke|story|poem)\b"
    ]
    for pattern in unsupported_patterns:
        if re.search(pattern, lower):
            return RoutingDecision(
                query_type="unsupported",
                should_retrieve=False,
                confidence=0.92,
                probabilities={"unsupported": 0.92, "document_question": 0.08},
                is_fallback=True,
                engine="fallback-heuristic",
                reason="Recognized out-of-scope non-document request."
            )

    # Default to document question
    return RoutingDecision(
        query_type="document_question",
        should_retrieve=True,
        confidence=0.85,
        probabilities={"document_question": 0.85, "unsupported": 0.15},
        is_fallback=True,
        engine="fallback-heuristic",
        reason="Query treated as document question for retrieval."
    )


# Comprehensive English stop words to accurately isolate query entities/topics
FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", 
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", 
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "tell", "give", "explain", "describe", "find", "book", "pdf"
}


def _fallback_context_sufficiency(query: str, retrieved_chunks: list[dict]) -> ContextSufficiencyDecision:
    """
    Conservative fallback that inspects semantic retrieval distance and query keyword presence
    to prevent hallucinations when OpenRouter Jev is offline or unconfigured.
    """
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", query.lower())
    significant_words = [w for w in tokens if w not in FALLBACK_STOP_WORDS]
    if not significant_words:
        significant_words = tokens

    combined_text = " ".join([c["text"].lower() for c in retrieved_chunks])
    distances = [c["distance"] for c in retrieved_chunks if c.get("distance") is not None]
    min_dist = min(distances) if distances else 1.0

    matched_words = [w for w in significant_words if w in combined_text]
    missing_words = [w for w in significant_words if w not in combined_text]
    overlap_ratio = len(matched_words) / max(len(significant_words), 1)

    # In Sentence Transformers (all-MiniLM-L6-v2) cosine distance:
    # min_dist > 1.25 indicates that even the best retrieved chunk has very poor semantic similarity.
    # If key entities are missing while distance > 1.15, context is judged insufficient.
    if min_dist > 1.25 or (len(missing_words) > 0 and min_dist > 1.15) or overlap_ratio < 0.25:
        quality = "insufficient"
        should_generate = False
        conf = 0.88
        probs = {"insufficient": 0.88, "sufficient": 0.04, "uncertain": 0.08}
        noul_val = 0.08
        reason = f"Context lacks required query entities (missing: {missing_words or 'low overlap'}) with high semantic distance ({min_dist:.2f})."
    elif (overlap_ratio >= 0.5 and min_dist <= 1.12) or min_dist < 0.95 or len(missing_words) == 0:
        quality = "sufficient"
        should_generate = True
        conf = 0.82
        probs = {"sufficient": 0.82, "insufficient": 0.08, "uncertain": 0.10}
        noul_val = 0.85
        reason = f"Retrieved chunks demonstrate strong relevance to query concepts (overlap: {overlap_ratio:.2f}, distance: {min_dist:.2f})."
    else:
        quality = "uncertain"
        should_generate = False  # Safe conservative path: avoid unsupported answers
        conf = 0.65
        probs = {"uncertain": 0.65, "sufficient": 0.15, "insufficient": 0.20}
        noul_val = 0.40
        reason = f"Context relevance is borderline (distance: {min_dist:.2f}); defaulting to safe non-generation."

    return ContextSufficiencyDecision(
        context_quality=quality,
        should_generate=should_generate,
        confidence=conf,
        probabilities=probs,
        noul_sufficiency=noul_val,
        is_fallback=True,
        engine="fallback-heuristic",
        reason=reason
    )
