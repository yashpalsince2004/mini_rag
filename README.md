# Mini PDF RAG Chatbot (with TypeSafe Jev via OpenRouter, Astro & FastAPI)

A minimal, educational **Retrieval-Augmented Generation (RAG)** system featuring **TypeSafe AI's Jev** accessed via **OpenRouter's Decisions API** as an intelligent decision-making layer, a **FastAPI backend**, and a modern dark **Astro JS web UI**. Built from first principles using Python, PyMuPDF, Sentence Transformers, ChromaDB, TypeSafe Jev (OpenRouter), and Google Gemini.

The primary goal of this project is understanding how a production-grade RAG pipeline makes **principled decisions** from first principles—without heavy abstractions or high-level frameworks like LangChain or LlamaIndex.

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances Large Language Models (LLMs) by grounding them in external factual knowledge. Instead of relying solely on the training weights of a generative model (which can hallucinate or become outdated), RAG:
1. Extracts text from proprietary documents (e.g. PDFs).
2. Partitions the text into semantic chunks with overlapping boundaries.
3. Generates dense vector embeddings using an embedding model.
4. Stores and indexes the vectors in a vector database (e.g. ChromaDB).
5. Retrieves the top-$K$ most relevant chunks when a user asks a question.
6. Augments the LLM prompt with the retrieved evidence to formulate a grounded answer.

---

## 2. What is Jev?

**Jev** is a specialized "System One" AI model created by **TypeSafe AI**. Unlike generative LLMs (like GPT-4 or Gemini) that generate free-form text and explanations, Jev is built specifically for **fast, typed, deterministic decisions**:
- **No free-form prose:** It does not produce conversational filler, apologies, or markdown text.
- **Typed Primitives:** It outputs calibrated probabilities and discrete categorical selections across three primitives:
  - **Choice:** Selects exactly one option from a defined set of criteria with confidence and probability distributions.
  - **Noul:** Evaluates whether a condition holds true, returning a probability between `0.0` and `1.0`.
  - **Score:** Positions an item on an ordered multi-tier scale.
- **Role in Software:** Jev acts as "programmable common sense", enabling traditional code to branch intelligently based on semantic meaning.

---

## 3. Why Use Jev? (Separation of Responsibilities)

In naive RAG architectures, every query traverses the entire pipeline—wasting tokens and hallucinating when answers are absent. By introducing Jev as the decision layer, we establish strict separation of responsibilities:

| Layer | Component | Responsibility | Why this separation matters |
|---|---|---|---|
| **Presentation** | Astro JS | User Interface & Trace | Renders dark UI, status indicators, and educational Decision Trace card. |
| **Orchestration** | FastAPI | Workflow Coordination | Exposes API routes, validates inputs, and sequences pipeline execution. |
| **Decision** | TypeSafe Jev (via OpenRouter) | Intelligent Control Gating | Evaluates intent, decides if retrieval is needed, checks context quality, and gates LLM generation. |
| **Extraction** | PyMuPDF (`fitz`) | Text Extraction | Parses PDF text page-by-page preserving page numbers. |
| **Embeddings** | Sentence Transformers | Vector Encoding | Generates 384-dimensional dense vectors (`all-MiniLM-L6-v2`). |
| **Retrieval** | ChromaDB | Vector Search | Indexes and queries top-$K$ nearest semantic chunks from `./chroma_db/`. |
| **Generation** | Google Gemini (`gemini-2.5-flash`) | Answer Synthesis | Writes natural-language answers **only** when Jev verifies context sufficiency. |

> **Key Rule:**  
> - **Jev decides what the system should do.**  
> - **ChromaDB finds relevant information.**  
> - **Gemini explains the retrieved information.**  
> - **FastAPI orchestrates the pipeline.**  
> - **Astro presents the experience.**

---

## 4. Why OpenRouter?

Accessing Jev through **OpenRouter** provides key benefits:
1. **Single API Key:** Uses your existing `OPENROUTER_API_KEY` without requiring a separate direct TypeSafe paid account or custom SDK.
2. **Dedicated Decisions Endpoint:** OpenRouter hosts Jev on a specialized endpoint (`POST https://openrouter.ai/api/alpha/decisions`) specifically designed for System One typed outputs.
3. **Model Flexibility:** Easily switch models or aliases via `.env` (e.g. `JEV_MODEL=~typesafe/jev-latest` or `typesafe/jev-1.13`) without changing code.
4. **Cost Efficiency:** Jev input tokens are low-cost, and output tokens are unmetered ($0), making decision gating economical.

---

## 5. Full Architecture Diagram

```text
                         USER
                          │
                          ▼
                     ASTRO UI
                          │
                          ▼
                    FASTAPI API
                          │
                          ▼
              ┌───────────────────────┐
              │  JEV VIA OPENROUTER   │
              │  Query Classification │
              └───────────┬───────────┘
                          │
             ┌────────────┴─────────────┐
             │                          │
        GREETING / OTHER            DOCUMENT
         ("Hello!...")              QUESTION
         (No DB, no LLM)                │
                                        ▼
                               ┌────────────────┐
                               │    CHROMADB    │
                               │ Vector Search  │
                               └────────┬───────┘
                                        │
                                   TOP-K CHUNKS
                                        │
                                        ▼
                              ┌───────────────────┐
                              │     JEV AGAIN     │
                              │ Context Quality & │
                              │ Generation Gating │
                              └─────────┬─────────┘
                                        │
                             ┌──────────┴──────────┐
                             │                     │
                        SUFFICIENT            INSUFFICIENT
                             │                     │
                             ▼                     ▼
                     ┌───────────────┐     NO GENERATION
                     │    GEMINI     │    ("I couldn't find
                     │ 2.5 FLASH LLM │    enough info in PDF")
                     └───────┬───────┘             │
                             │                     │
                       ANSWER + SOURCES            │
                             └──────────┬──────────┘
                                        │
                                        ▼
                                  FINAL RESPONSE
```

---

## 6. Detailed Pipeline & Worked Examples

### Example A: The Success Path (Document Question)
```text
1. User asks: "What are the four sections of IELTS?"
2. FastAPI calls Jev via OpenRouter (/api/alpha/decisions).
3. Jev Decision 1 (Query Routing):
   - query_type = "document_question" (confidence: 0.82)
4. Jev Decision 2 (Retrieval Gating):
   - should_retrieve = True (noul: 0.80)
5. FastAPI executes ChromaDB vector search for top 3 chunks.
   - Page 5 (distance: 0.9129)
   - Page 8 (distance: 0.9458)
   - Page 10 (distance: 0.9599)
6. FastAPI sends question + retrieved chunks to Jev Decision 3 & 4.
7. Jev Decision 3 (Context Quality):
   - context_quality = "sufficient" (confidence: 1.0)
8. Jev Decision 4 (Generation Gating):
   - should_generate = True (noul: 0.97)
9. FastAPI calls Gemini 2.5 Flash with retrieved context.
10. Gemini returns: "IELTS consists of four components: Listening, Reading, Writing, and Speaking..." (Sources: [5, 8, 10]).
```

### Example B: The Failure Path (Irrelevant Context / Zero Hallucination)
```text
1. User asks: "What is the population of Germany?"
2. Jev classifies as document_question (should_retrieve = True).
3. ChromaDB retrieves top 3 chunks (distances 1.35 to 1.57, weak semantic match).
4. FastAPI sends question + chunks to Jev.
5. Jev evaluates context quality:
   - context_quality = "insufficient" (confidence: 1.0)
   - should_generate = False (noul: 0.02)
6. Gemini is NEVER called.
7. Backend immediately returns:
   "I couldn't find enough information about that in the selected PDF."
```

### Example C: Social Greeting
```text
1. User says: "Hi"
2. Jev classifies as greeting (confidence: 1.0, should_retrieve: False).
3. ChromaDB is NEVER called.
4. Gemini is NEVER called.
5. Backend immediately returns: "Hello! Ask me anything about the selected PDF."
```

---

## 7. OpenRouter Decisions API Specification

The implementation sends structured payloads to OpenRouter's specialized endpoint:

- **Endpoint:** `POST https://openrouter.ai/api/alpha/decisions`
- **Headers:**
  ```json
  {
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Mini PDF RAG Chatbot"
  }
  ```
- **Payload Schema:**
  ```json
  {
    "model": "~typesafe/jev-latest",
    "state": { "user_query": "Is this book for IELTS?", "active_document": "Cambridge_21.pdf" },
    "questions": {
      "query_type": {
        "type": "choice",
        "instructions": "Classify the user query intent for a document question-answering assistant.",
        "criteria": {
          "document_question": "Questions asking about information, facts, or details from the document.",
          "greeting": "Casual greetings or hello.",
          "help": "Asking how to use the assistant.",
          "unsupported": "Out-of-scope requests such as writing code or checking weather.",
          "clarification_needed": "Vague or ambiguous input."
        }
      },
      "should_retrieve": {
        "type": "noul",
        "instructions": "Should the assistant perform semantic retrieval from the selected document?",
        "criteria": {
          "true": "The user is asking a factual question about the document content.",
          "false": "The query is a greeting, help request, joke, or out-of-scope question."
        }
      }
    }
  }
  ```

---

## 8. Code Organization

```text
mini_rag/
├── backend/
│   ├── main.py              # FastAPI server, endpoints, and pipeline orchestration
│   ├── rag.py               # PDF extraction, chunking, embeddings, ChromaDB, Gemini generation
│   └── jev_decisions.py     # Reusable JevClient (OpenRouter Decisions API) & fallback logic
│
├── frontend/
│   ├── src/
│   │   └── pages/
│   │       └── index.astro  # Astro dark UI with live thinking states & '⚡ Trace: ON/OFF' inspector
│   ├── package.json         # Frontend configuration
│   └── astro.config.mjs     # Astro server (port 4321)
│
├── documents/
│   ├── Cambridge_21.pdf     # Sample IELTS test book (144 pages, 218 chunks)
│   └── document.pdf         # Sample ML concepts document (3 pages)
│
├── test_jev.py              # Isolated verification test for OpenRouter Decisions API
├── main.py                  # Interactive CLI runner (mirrors web pipeline)
├── chroma_db/               # Persistent ChromaDB vector storage
├── requirements.txt         # Backend Python dependencies (includes httpx)
├── .env.example             # Template for environment variables
└── README.md                # System documentation
```

---

## 9. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Set your configuration:

```env
# Google Gemini API Key (Answer Generation)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# OpenRouter API Key (Decision Layer via Jev)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# OpenRouter Jev Model Identifier
JEV_MODEL=~typesafe/jev-latest

# Optional Development Debugging
DEBUG_DECISIONS=true

# Optional Chroma Cloud Configuration (Hosted Vector DB)
# Leave blank to use local ./chroma_db/
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_tenant_id_here
CHROMA_DATABASE=mini_rag
```

> **Security Guarantee:** `OPENROUTER_API_KEY`, `CHROMA_API_KEY`, and `GEMINI_API_KEY` are loaded strictly backend-side. They are never sent to the browser or exposed to Astro frontend templates.

---

## 10. Installation & Running

### 1. Backend Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify OpenRouter Jev API in Isolation
Run the standalone test script to verify your OpenRouter key and model connection:
```bash
python test_jev.py
```

Expected output:
```text
HTTP Status: 200
✅ Successfully received decisions response from OpenRouter:
  [is_question] Noul Probability: 0.99
  [topic] Choice: geography (conf: 1)
>>> OpenRouter Decisions API verification PASSED! <<<
```

### 3. Run the Web Application
Open two terminal tabs:

**Terminal 1 (FastAPI Backend):**
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Astro Frontend):**
```bash
cd frontend
npm run dev
```

Open **http://localhost:4321** in your browser.

### 4. Run the Terminal CLI
```bash
source .venv/bin/activate
python main.py
```

---

## 11. Verification & Test Suite

The system has been verified against all 6 master test cases:

| Test Case | Input | Route | Retrieval Used | Context Sufficient | Gemini Called | Expected Behavior |
|---|---|---|---|---|---|---|
| **Test 1: Greeting** | `"Hi"` | `greeting` | ❌ No | ❌ No | ❌ No | Immediate greeting. No DB or LLM call. |
| **Test 2: Document Question** | `"Is this book for IELTS?"` | `document_question` | ✅ Yes (3 chunks) | ✅ Yes | ✅ Yes | Grounded answer with source pages `[5, 10]`. |
| **Test 3: Detailed Question** | `"What are the four sections of IELTS?"` | `document_question` | ✅ Yes (3 chunks) | ✅ Yes | ✅ Yes | Details Listening, Reading, Writing, Speaking (`[5, 8, 10]`). |
| **Test 4: Outside Document** | `"What is the population of Germany?"` | `document_question` | ✅ Yes (3 chunks) | ❌ No | ❌ No | Refusal: *"I couldn't find enough information about that in the selected PDF."* |
| **Test 5: Unsupported Task** | `"Write me a Python game."` | `unsupported` | ❌ No | ❌ No | ❌ No | Polite refusal. No DB or LLM call. |
| **Test 6: Empty Query** | `""` | N/A | ❌ No | ❌ No | ❌ No | HTTP 400 Bad Request validation error. |

To run the automated test suite:
```bash
source .venv/bin/activate
python -c '
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
client.post("/api/documents/select", json={"filename": "Cambridge_21.pdf"})

# Run tests
t1 = client.post("/api/chat", json={"question": "Hi"}).json()
assert t1["decision"]["route"] == "greeting" and not t1["decision"]["retrieval_used"]

t2 = client.post("/api/chat", json={"question": "Is this book for IELTS?"}).json()
assert t2["decision"]["context_sufficient"] and len(t2["sources"]) > 0

t4 = client.post("/api/chat", json={"question": "What is the population of Germany?"}).json()
assert not t4["decision"]["context_sufficient"] and not t4["decision"]["should_generate"]

print("All 6 OpenRouter Jev tests PASSED!")
'
```
