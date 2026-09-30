# Mini PDF RAG Chatbot (with TypeSafe Jev, Astro & FastAPI)

A minimal, educational **Retrieval-Augmented Generation (RAG)** system featuring **TypeSafe AI's Jev** as an intelligent decision-making layer, a **FastAPI backend**, and a modern dark **Astro JS web UI**. Built from scratch using Python, PyMuPDF, Sentence Transformers, ChromaDB, TypeSafe Jev, and Google Gemini.

The primary goal of this project is understanding how a production-grade RAG pipeline makes **principled decisions** from first principles—without heavy abstractions or high-level frameworks like LangChain or LlamaIndex.

---

## 1. Project Overview & Responsibility Matrix

Every component in this architecture has a single, strictly separated responsibility:

| Component | Responsibility | Why this separation is intentional |
|---|---|---|
| **Astro** | User interface & Presentation | Modern, responsive dark UI with live thinking indicators and an educational Jev Decision Trace panel. |
| **FastAPI** | Pipeline Orchestration | Handles API endpoints, input validation, and coordinates the flow between Jev, ChromaDB, and Gemini. |
| **TypeSafe Jev** | Decision Layer | Intelligently evaluates user intent, controls when retrieval occurs, validates context sufficiency, and gates LLM generation. |
| **PyMuPDF (`fitz`)** | PDF Text Extraction | Extracts raw text page-by-page while preserving accurate page metadata. |
| **Sentence Transformers** | Dense Vector Embeddings | Converts text passages into 384-dimensional dense semantic vectors using `all-MiniLM-L6-v2`. |
| **ChromaDB** | Vector Similarity Retrieval | Indexes and queries top-$K$ nearest semantic chunks using persistent local storage (`./chroma_db/`). |
| **Google Gemini (`gemini-2.5-flash`)** | Grounded Answer Generation | Synthesizes natural-language answers strictly grounded in retrieved evidence—invoked **only** when Jev verifies context sufficiency. |

> **Key Architectural Principle:**  
> - **Jev decides what the system should do.**  
> - **ChromaDB finds relevant information.**  
> - **Gemini explains the retrieved information.**  
> - **FastAPI orchestrates the pipeline.**  
> - **Astro presents the experience.**

---

## 2. The Jev Decision Layer

### Why Jev was Introduced

In traditional naive RAG architectures, every user input blindly traverses the entire pipeline:

```text
Naive RAG (Before):

User Query
    │
    ▼
Embed Query (Sentence Transformers)
    │
    ▼
ChromaDB Vector Search (Top-K Chunks)
    │
    ▼
Gemini LLM Generation
    │
    ▼
Final Answer (often hallucinated or apologizing)
```

#### Flaws in Naive RAG:
1. **Unnecessary Retrieval & LLM Calls:** If a user says `"Hi"`, naive RAG embeds the greeting, searches ChromaDB for random nearest chunks, and asks Gemini to formulate a response.
2. **Hallucination on Missing Context:** If a user asks an out-of-document question (e.g., *"What is the population of Germany?"* when reading an IELTS test preparation book), ChromaDB will still return its top 3 closest chunks. Because distances are relative, Gemini is prompted with irrelevant snippets and may hallucinate or formulate an unverified response.
3. **Arbitrary Distance Thresholds:** Hard-coding rules like `if distance < 1.0` is fragile across different embedding models and document domains.

---

### The New Architecture (After Jev Integration)

```text
Decision-Gated RAG (After):

                         USER
                          │
                          ▼
                     ASTRO UI
                          │
                          ▼
                    FASTAPI API
                          │
                          ▼
                   ┌─────────────┐
                   │  JEV ROUTER │
                   └──────┬──────┘
                          │
             ┌────────────┼─────────────┐
             │            │             │
             ▼            ▼             ▼
         GREETING    DOC QUESTION   UNSUPPORTED
      ("Hello!...")       │         ("I only answer
                          ▼          doc questions")
                      CHROMADB
                      RETRIEVER
                          │
                          ▼
                     TOP-K CHUNKS
                          │
                          ▼
                   ┌─────────────┐
                   │ JEV CONTEXT │
                   │  EVALUATION │
                   └──────┬──────┘
                          │
                ┌─────────┴─────────┐
                │                   │
           SUFFICIENT          INSUFFICIENT
                │                   │
                ▼                   ▼
             GEMINI            NO GENERATION
            GENERATOR      ("I couldn't find enough
                │          information in the PDF.")
                ▼                   │
          ANSWER + SOURCES          │
                └─────────┬─────────┘
                          │
                          ▼
                   FINAL RESPONSE
```

---

### The Four Jev Decision Stages

Rather than creating a single monolithic decision, the decision layer is split into focused, typed stages:

#### Stage 1: Query Intent Routing (`classify_query_intent`)
Classifies user intent into discrete, typed choices:
- `document_question`: Specific inquiry regarding content in the document.
- `greeting`: Social greeting or conversational opener (`"Hi"`, `"Hello"`).
- `help`: Request for instructions or capabilities (`"How do I use this?"`).
- `unsupported`: Out-of-scope requests (`"Write me a Python game"`, `"Tell me a joke"`).
- `clarification_needed`: Vague, incomplete, or ambiguous inputs.

#### Stage 2: Retrieval Gating (`should_retrieve`)
A typed boolean decision determining whether ChromaDB retrieval should execute:
- `"Is this book for IELTS?"` → `should_retrieve = True`
- `"Hi"` → `should_retrieve = False` (Direct greeting response returned immediately)
- `"Write a snake game"` → `should_retrieve = False` (Direct refusal returned immediately)

#### Stage 3: Context Quality Assessment (`evaluate_context_sufficiency`)
After ChromaDB returns the top-$K$ chunks with their semantic distances and page metadata, Jev evaluates whether the retrieved text actually contains the necessary factual evidence:
- `sufficient`: The retrieved chunks explicitly state facts that answer the question.
- `insufficient`: The retrieved chunks do not contain enough information or are only tangentially related.
- `uncertain`: The chunks partially touch upon the topic but miss crucial facts.

#### Stage 4: Generation Control (`should_generate`)
A Noul probability decision controlling whether Gemini is permitted to run:
- If `context_quality == "sufficient"` → `should_generate = True` → Gemini is invoked.
- If `context_quality == "insufficient"` → `should_generate = False` → Gemini is blocked. The backend returns:
  > *"I couldn't find enough information about that in the selected PDF."*
- If `context_quality == "uncertain"` → conservative fallback: Gemini is blocked to prevent hallucinations.

---

### Official TypeSafe SDK Implementation

The project uses the official `typesafe-sdk` (`v0.7.2`). Decisions are declared with `Choice` and `Noul`:

```python
from typesafe import TypeSafeClient, Choice, Noul, NoulCriteria

# Decision 1: Query Routing
ROUTING_QUESTIONS = {
    "query_type": Choice(
        instructions="Classify the user query into the single most accurate category.",
        criteria={
            "document_question": "A question asking for information, facts, or explanations from the document.",
            "greeting": "A conversational greeting such as 'hello', 'hi', or 'good morning'.",
            "help": "A request for help or instructions on using the application.",
            "unsupported": "A request unrelated to documents, such as asking to write code, tell jokes, or current weather.",
            "clarification_needed": "A query that is too vague, fragmented, or ambiguous to understand."
        }
    ),
    "should_retrieve": Noul(
        instructions="Does answering this query require searching the document vector database?",
        criteria=NoulCriteria(
            true="The query is a factual question about the document and requires retrieval.",
            false="The query is a greeting, help request, joke, or out-of-scope task."
        )
    )
}

# Decision 2: Context Quality & Generation Gating
CONTEXT_QUESTIONS = {
    "context_quality": Choice(
        instructions="Based on the retrieved document chunks, does the context contain sufficient factual information to answer the question accurately?",
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
            false="The retrieved chunks lack necessary facts; answering would require guessing or hallucinating."
        )
    )
}
```

---

### Performance & Safety Principles

1. **Elimination of Unnecessary LLM Invocations:** Greetings, help queries, and unrelated tasks bypass both embedding calculations, ChromaDB lookups, and Gemini calls.
2. **Anti-Hallucination Gating:** Gemini is never prompted with irrelevant context. If the document doesn't contain the answer, Jev stops the pipeline before generation occurs.
3. **Graceful Fallback:** If `TYPESAFE_API_KEY` is not configured or the network is unavailable, the backend employs a conservative local fallback heuristic based on lexical entity verification and cosine distance boundaries.

---

## 3. Code Organization

The codebase is organized into small, educational modules with clear responsibilities:

```text
mini_rag/
├── backend/
│   ├── main.py              # FastAPI server, endpoints, and pipeline orchestration
│   ├── rag.py               # PDF extraction, chunking, embeddings, ChromaDB, Gemini generation
│   └── jev_decisions.py     # TypeSafe Jev decisions (routing, retrieval gating, context quality)
│
├── frontend/
│   ├── src/
│   │   └── pages/
│   │       └── index.astro  # Astro dark-theme UI with Decision Trace toggle
│   ├── package.json         # Frontend configuration
│   └── astro.config.mjs     # Astro server (port 4321)
│
├── documents/
│   ├── Cambridge_21.pdf     # Sample IELTS test book (144 pages, 218 chunks)
│   └── document.pdf         # Sample ML concepts document (3 pages)
│
├── main.py                  # Interactive CLI runner (mirrors web pipeline)
├── chroma_db/               # Persistent ChromaDB vector storage
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md                # System documentation
```

### Module Responsibilities:
- **`backend/main.py`**: Declares FastAPI routes (`/api/health`, `/api/documents`, `/api/documents/select`, `/api/chat`). Coordinates the Jev routing step, conditional ChromaDB query, Jev context sufficiency step, and Gemini generation.
- **`backend/rag.py`**: Houses core RAG mechanics: `extract_pdf_text` (PyMuPDF), `create_chunks` (word-based chunking with overlap), `ingest_document`, `retrieve_relevant_chunks`, and `generate_grounded_answer`.
- **`backend/jev_decisions.py`**: Manages `TypeSafeClient`, defines `Choice` and `Noul` questions, formats evaluation state, extracts confidence/probabilities, and provides conservative fallback logic.

---

## 4. Frontend & Backend API Flow

The browser communicates strictly over HTTP with FastAPI:

1. **`GET /api/documents`**: Discovers available `.pdf` files in `documents/`.
2. **`POST /api/documents/select`**: Validates filename, checks ChromaDB collection cache, and indexes chunks if needed.
3. **`POST /api/chat`**:
   - Request: `{"question": "Is this book for IELTS?", "filename": "Cambridge_21.pdf"}`
   - Response:
     ```json
     {
       "answer": "Yes, this book contains authentic examination papers for IELTS preparation...",
       "sources": [5, 10],
       "decision": {
         "query_type": "document_question",
         "should_retrieve": true,
         "should_generate": true,
         "context_quality": "sufficient",
         "confidence": 0.94,
         "noul_sufficiency": 0.91,
         "engine": "typesafe-jev (jev-latest)",
         "is_fallback": false,
         "retrieved_count": 3,
         "top_page": 5,
         "top_distance": 1.0147
       }
     }
     ```

> **Security Guarantee:** `TYPESAFE_API_KEY` and `GEMINI_API_KEY` remain backend-side only. They are never sent to the Astro frontend or browser client.

---

## 5. Educational Decision Trace Mode

The Astro frontend features a dedicated **⚡ Trace: ON/OFF** toggle in the chat header.

When enabled, each assistant response displays a sleek decision badge and collapsible trace card showing:
- **Query Classification:** `document_question`, `greeting`, etc.
- **Retrieval Action:** Chunks searched vs. bypassed.
- **Context Quality Badge:** `sufficient`, `insufficient`, or `uncertain`.
- **Generation Permission:** `Allowed` vs. `Blocked`.
- **Top Result Metrics:** Page number, semantic distance, and chunk count.

When disabled, users enjoy a distraction-free, polished conversational experience with subtle live status text (*"Evaluating query intent..."*, *"Searching ChromaDB..."*, *"Verifying context sufficiency..."*).

---

## 6. Installation & Setup

### 1. Python Environment (Backend)
Requires Python 3.10+ (Python 3.12 recommended):

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (includes typesafe-sdk, chromadb, google-genai, fastapi)
pip install -r requirements.txt
```

### 2. Node.js Environment (Frontend)
Requires Node 18+:

```bash
cd frontend
npm install
cd ..
```

### 3. Environment Variables
Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Configure your API keys:

```env
# Google Gemini API Key (Required for natural language answer generation)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# TypeSafe AI Jev API Key (Required for Jev decision-making layer)
TYPESAFE_API_KEY=your_typesafe_api_key_here
TYPESAFE_MODEL=jev-latest
```

*(Note: If `TYPESAFE_API_KEY` is not provided, the application automatically runs in conservative fallback mode without crashing).*

---

## 7. Running the Application

### Option A: Web Application (Astro + FastAPI)

Open two terminal tabs:

**Terminal 1 (Backend API):**
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000
```
*Backend runs on: `http://localhost:8000`*

**Terminal 2 (Frontend UI):**
```bash
cd frontend
npm run dev
```
*Frontend runs on: `http://localhost:4321`*

Open **http://localhost:4321** in your browser.

---

### Option B: Terminal CLI Runner

Run the interactive terminal interface directly:

```bash
source .venv/bin/activate
python main.py
```

---

## 8. Verification & Test Suite

The system has been verified against all 6 master test cases:

| Test Case | Query | Jev Intent | ChromaDB Retrieval | Jev Context Quality | Gemini Generation | Result |
|---|---|---|---|---|---|---|
| **Test 1: Greeting** | `"Hi"` | `greeting` | ❌ Bypassed | N/A | ❌ Bypassed | Returns greeting immediately. No DB or LLM call. |
| **Test 2: Document Question** | `"Is this book for IELTS?"` | `document_question` | ✅ Retrieved (3 chunks) | `sufficient` | ✅ Allowed | Answers with source pages `[5, 10]`. |
| **Test 3: Specific Question** | `"What are the four components of IELTS?"` | `document_question` | ✅ Retrieved (3 chunks) | `sufficient` | ✅ Allowed | Explains Listening, Reading, Writing, Speaking (`[5, 8, 10]`). |
| **Test 4: Out-of-Document** | `"What is the population of Germany?"` | `document_question` | ✅ Retrieved (3 chunks) | `insufficient` | ❌ Blocked | Returns: *"I couldn't find enough information about that in the selected PDF."* |
| **Test 5: Unsupported Task** | `"Write me a Python game."` | `unsupported` | ❌ Bypassed | N/A | ❌ Bypassed | Refuses politely. No DB or LLM call. |
| **Test 6: Empty Query** | `""` | N/A | ❌ Bypassed | N/A | ❌ Bypassed | HTTP 400 Bad Request validation error. |

To run the automated verification script:

```bash
source .venv/bin/activate
python -c '
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
client.post("/api/documents/select", json={"filename": "Cambridge_21.pdf"})

# Run tests
t1 = client.post("/api/chat", json={"question": "Hi", "filename": "Cambridge_21.pdf"}).json()
assert t1["decision"]["query_type"] == "greeting" and t1["decision"]["should_retrieve"] == False

t2 = client.post("/api/chat", json={"question": "Is this book for IELTS?", "filename": "Cambridge_21.pdf"}).json()
assert t2["decision"]["context_quality"] == "sufficient" and t2["decision"]["should_generate"] == True

t4 = client.post("/api/chat", json={"question": "What is the population of Germany?", "filename": "Cambridge_21.pdf"}).json()
assert t4["decision"]["context_quality"] == "insufficient" and t4["decision"]["should_generate"] == False
print("Verification complete: All tests passed!")
'
```

---

## 9. Known Limitations

- **Single-Turn Scope:** The backend processes each question independently without an external conversational memory database. The frontend preserves client-side chat bubbles for continuity. If a user asks *"What is IELTS?"* followed by *"What are its four components?"*, the second question is embedded directly.
- **Text-Only PDFs:** The pipeline extracts text via PyMuPDF. Scanned PDFs containing only bitmap images require an external OCR pre-processing step.
