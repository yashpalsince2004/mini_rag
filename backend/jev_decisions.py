import os
import re
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

try:
    from typesafe_sdk import Choice, Noul, NoulCriteria, TypeSafeClient
    from typesafe_sdk._core.errors import TypeSafeError, TypeSafeAPIError
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    TYPESAFE_SDK_AVAILABLE = False
    Choice, Noul, NoulCriteria, TypeSafeClient = None, None, None, None
    TypeSafeError, TypeSafeAPIError = Exception, Exception

TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")


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
    engine: str = "typesafe-jev"
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
    engine: str = "typesafe-jev"
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# Jev Client Factory
# =====================================================================

def get_typesafe_client() -> Optional[TypeSafeClient]:
    """
    Returns an initialized TypeSafeClient if TYPESAFE_API_KEY is available.
    Returns None if not configured or SDK is unavailable.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key or not TYPESAFE_SDK_AVAILABLE:
        return None

    try:
        return TypeSafeClient(api_key=api_key.strip(), model=TYPESAFE_MODEL)
    except Exception as e:
        print(f"[JEV] Warning: Failed to initialize TypeSafeClient ({e}). Falling back.")
        return None


# =====================================================================
# Decision 1 & 2: Query Routing and Retrieval Gating
# =====================================================================

ROUTING_QUESTIONS = {
    "query_type": Choice(
        instructions="Classify the user query intent for a document question-answering assistant.",
        criteria={
            "document_question": "Questions asking about information, facts, concepts, or details from the document or PDF.",
            "greeting": "Casual greetings, hellos, good mornings, or introductory pleasantries.",
            "help": "Asking how to use this tool, what it does, or asking for instructions on using the assistant.",
            "unsupported": "Out-of-scope requests such as writing software, generating creative stories, checking real-time weather, or general knowledge unrelated to documents.",
            "clarification_needed": "Empty, garbled, or completely ambiguous input that cannot be interpreted without clarification."
        }
    )
} if TYPESAFE_SDK_AVAILABLE else {}


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

    client = get_typesafe_client()

    if client:
        try:
            print(f"[JEV] Invoking System One model ({TYPESAFE_MODEL}) for query classification...")
            state = {
                "user_query": trimmed,
                "active_document": filename or "No document loaded"
            }
            response = client.system_one(
                state=state,
                questions=ROUTING_QUESTIONS,
                model=TYPESAFE_MODEL
            )

            answer = response.answers["query_type"]
            chosen = answer.choice
            conf = float(answer.confidence)
            probs = {k: float(v) for k, v in answer.probabilities.items()}

            should_retrieve = (chosen == "document_question")

            print(f"[JEV] Query classification: {chosen} (confidence: {conf:.2f})")
            print(f"[JEV] Retrieval decision: {'yes' if should_retrieve else 'no'}")

            return RoutingDecision(
                query_type=chosen,
                should_retrieve=should_retrieve,
                confidence=conf,
                probabilities=probs,
                is_fallback=False,
                engine=f"typesafe-jev ({TYPESAFE_MODEL})",
                reason=f"Jev classified query as {chosen} with {conf:.2f} confidence."
            )
        except Exception as e:
            print(f"[JEV] Warning: TypeSafe API call failed ({e}). Using conservative fallback.")

    # Conservative Heuristic Fallback when Jev is offline / unconfigured
    return _fallback_query_routing(trimmed)


def _fallback_query_routing(query: str) -> RoutingDecision:
    """Deterministic fallback that mimics Jev's routing criteria when API key is unset."""
    lower = query.lower().strip()
    clean = re.sub(r'[^\w\s]', '', lower)

    # Greeting check
    greeting_tokens = {"hi", "hello", "hey", "hiya", "howdy", "good morning", "good evening", "good afternoon"}
    if clean in greeting_tokens or clean.startswith("hello ") or clean.startswith("hi "):
        print(f"[JEV-FALLBACK] Query classification: greeting (confidence: 0.95)")
        print(f"[JEV-FALLBACK] Retrieval decision: no")
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
        print(f"[JEV-FALLBACK] Query classification: help (confidence: 0.90)")
        print(f"[JEV-FALLBACK] Retrieval decision: no")
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
            print(f"[JEV-FALLBACK] Query classification: unsupported (confidence: 0.92)")
            print(f"[JEV-FALLBACK] Retrieval decision: no")
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
    print(f"[JEV-FALLBACK] Query classification: document_question (confidence: 0.85)")
    print(f"[JEV-FALLBACK] Retrieval decision: yes")
    return RoutingDecision(
        query_type="document_question",
        should_retrieve=True,
        confidence=0.85,
        probabilities={"document_question": 0.85, "unsupported": 0.15},
        is_fallback=True,
        engine="fallback-heuristic",
        reason="Query treated as document question for retrieval."
    )


# =====================================================================
# Decision 3 & 4: Retrieval Quality and Generation Gating
# =====================================================================

CONTEXT_QUESTIONS = {
    "context_quality": Choice(
        instructions="Based on the retrieved document chunks, does the context contain sufficient factual information to answer the user's question accurately?",
        criteria={
            "sufficient": "The retrieved chunks explicitly state facts, details, or explanations that directly address the question.",
            "insufficient": "The retrieved chunks do not contain enough information to answer the question, or are only tangentially related.",
            "uncertain": "The chunks partially touch upon the topic but are missing crucial facts or leave the answer ambiguous."
        }
    ),
    "should_generate": Noul(
        instructions="Should the answering model be allowed to generate a factual answer based on these retrieved chunks?",
        criteria=NoulCriteria(
            true="The retrieved chunks contain verifiable evidence to answer the question without hallucination.",
            false="The retrieved chunks lack the necessary facts, so generating an answer would require guessing or hallucinating."
        )
    )
} if TYPESAFE_SDK_AVAILABLE else {}


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
        print("[JEV] Context quality: insufficient (no chunks retrieved)")
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

    # Format state for Jev
    client = get_typesafe_client()
    if client:
        try:
            print(f"[JEV] Invoking System One model ({TYPESAFE_MODEL}) for context sufficiency assessment...")
            state = {
                "question": query,
                "document": filename or "Selected document",
                "retrieved_chunks": [
                    {
                        "chunk_index": i + 1,
                        "page": chunk["page"],
                        "distance": round(chunk["distance"], 4) if chunk.get("distance") is not None else None,
                        "text": chunk["text"]
                    }
                    for i, chunk in enumerate(retrieved_chunks)
                ]
            }

            response = client.system_one(
                state=state,
                questions=CONTEXT_QUESTIONS,
                model=TYPESAFE_MODEL
            )

            choice_ans = response.answers["context_quality"]
            noul_ans = response.answers["should_generate"]

            chosen_quality = choice_ans.choice
            conf = float(choice_ans.confidence)
            probs = {k: float(v) for k, v in choice_ans.probabilities.items()}
            noul_prob = float(noul_ans.noul)

            # Generation rule: allow generation if sufficient, or if uncertain with high noul (>0.6)
            should_generate = (chosen_quality == "sufficient") or (chosen_quality == "uncertain" and noul_prob >= 0.65)

            print(f"[JEV] Context quality: {chosen_quality} (confidence: {conf:.2f}, noul_prob: {noul_prob:.2f})")
            print(f"[JEV] Generation decision: {'yes' if should_generate else 'no'}")

            return ContextSufficiencyDecision(
                context_quality=chosen_quality,
                should_generate=should_generate,
                confidence=conf,
                probabilities=probs,
                noul_sufficiency=noul_prob,
                is_fallback=False,
                engine=f"typesafe-jev ({TYPESAFE_MODEL})",
                reason=f"Context evaluated as {chosen_quality} (noul probability: {noul_prob:.2f})."
            )
        except Exception as e:
            print(f"[JEV] Warning: TypeSafe API call failed ({e}). Using conservative fallback.")

    # Conservative Heuristic Fallback when Jev is offline / unconfigured
    return _fallback_context_sufficiency(query, retrieved_chunks)


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
    to prevent hallucinations when Jev is offline or unconfigured.
    """
    # Extract significant query words (ignoring grammatical stop words)
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

    print(f"[JEV-FALLBACK] Context quality: {quality} (overlap: {overlap_ratio:.2f}, min_dist: {min_dist:.2f})")
    print(f"[JEV-FALLBACK] Generation decision: {'yes' if should_generate else 'no'}")

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
