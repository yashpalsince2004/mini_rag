# Mini PDF RAG Chatbot (with Astro & FastAPI)

A minimal, educational **Retrieval-Augmented Generation (RAG)** system featuring a **FastAPI backend** and a modern, dark **Astro JS web UI**. Built from scratch using Python, PyMuPDF, Sentence Transformers, ChromaDB, and Google Gemini.

The primary goal of this project is learning and understanding how the internal RAG pipeline works from first principles—without heavy abstractions or high-level frameworks such as LangChain or LlamaIndex.

---

## 1. Project Overview & Division of Responsibilities

- **Astro JS:** Serves solely as the lightweight, modern user interface. It never communicates directly with ChromaDB, Gemini, or local files.
- **FastAPI:** Acts as the backend API layer exposing endpoints for document discovery, ingestion, and RAG question-answering.
- **PyMuPDF (`fitz`):** Extracts raw text page-by-page from local PDF files while preserving page numbers.
- **Sentence Transformers (`all-MiniLM-L6-v2`):** Encodes text passages into dense 384-dimensional vector embeddings.
- **ChromaDB:** Stores vectors, original chunk text, and page metadata in a persistent local database (`./chroma_db/`).
- **Google Gemini API (`gemini-2.5-flash`):** Synthesizes accurate, natural-language answers strictly grounded in the retrieved context.

---

## 2. Complete Architecture Diagram

```text
                 Astro Frontend (http://localhost:4321)
                       │
                       │ HTTP / JSON
                       ▼
                 FastAPI Backend (http://localhost:8000)
                       │
              ┌────────┴────────┐
              ▼                 ▼
        PDF Processing       User Query
              │                 │
          PyMuPDF           Embedding (all-MiniLM-L6-v2)
              │                 │
          Chunking          ChromaDB Similarity Search
              │                 │
        Embeddings              ▼
              │             Top-K Chunks + Distances
              ▼                 │
          ChromaDB              ▼
                    Gemini 2.5 Flash
                              │
                              ▼
                           Answer
                              │
                              ▼
                       Astro Frontend (UI Display + Sources)
```

---

## 3. Frontend & Backend Communication

The browser communicates strictly over HTTP with the FastAPI server:

1. **Document Discovery:** `GET /api/documents` retrieves the list of `.pdf` files residing in `documents/`.
2. **Document Ingestion:** `POST /api/documents/select` validates the file path, parses the PDF, creates chunks, computes embeddings, and indexes them in ChromaDB (reusing the collection if already indexed).
3. **Chat Interaction:** `POST /api/chat` embeds the user question, queries ChromaDB for the top-$K$ nearest chunks, augments the Gemini prompt with retrieved context, and returns the grounded answer with source pages.

---

## 4. PDF Selection Flow

```text
User opens Web UI
       │
       ▼
Frontend calls GET /api/documents
       │
       ▼
FastAPI scans documents/ directory (only .pdf files, path traversal blocked)
       │
       ▼
Dropdown populates with available files (e.g. Cambridge_21.pdf, document.pdf)
       │
       ▼
User clicks [ Load Document ]
       │
       ▼
Frontend calls POST /api/documents/select
```

---

## 5. PDF Ingestion Flow

When a PDF is selected:

1. **Path Validation:** FastAPI ensures the file exists in `documents/` and blocks `../` directory traversal.
2. **Collection Check:** ChromaDB checks if a collection for this document already exists (e.g., `pdf_cambridge_21`). If valid, it immediately returns without re-embedding.
3. **Text Extraction:** PyMuPDF parses the document page-by-page:
   ```python
   # 1. Extract PDF text
   doc = fitz.open(pdf_path)
   for page_index in range(len(doc)):
       text = doc[page_index].get_text().strip()
   ```
4. **Chunking:** The extracted text is partitioned into overlapping word chunks.
5. **Vector Embedding:** `all-MiniLM-L6-v2` converts chunks into 384-dimensional vectors.
6. **Storage:** Vectors, raw text, and page metadata are written to ChromaDB.

---

## 6. Text Chunking

Text cannot be embedded as one giant document because:
- LLM and embedding context windows are finite.
- Embedding an entire book into one vector dilutes specific factual details.
- Similarity search works best against concise, topical passages.

### Sliding-Window Configuration
- **`chunk_size = 300` words:** The target length of each discrete chunk.
- **`chunk_overlap = 50` words:** The number of words repeated from the previous chunk.

```text
Chunk 1: [Words 1 to 300]
                │
                ◄── 50-word Overlap ──►
                │
Chunk 2:       [Words 251 to 550]
```

**Why Overlap Matters:** Overlapping guarantees that ideas and sentences spanning a 300-word boundary are not cut off.

---

## 7. Vector Embeddings

An embedding model maps textual meaning to numerical coordinates in 384-dimensional space:

```text
"Supervised learning algorithms use labeled data."
                        ↓
[ 0.0214, -0.1820, 0.4431, -0.0092, ... 384 coordinates ]
```

### The Golden Rule of Vector Search
> **The exact same embedding model (`all-MiniLM-L6-v2`) must be used for document chunks and user queries.**

Because all vectors reside in the identical geometric coordinate system, cosine distances accurately reflect semantic similarity.

---

## 8. ChromaDB Storage & Document-Specific Collections

ChromaDB runs locally in `./chroma_db/`.

To prevent chunks from different PDFs from colliding, each PDF is assigned a deterministic collection name:
- `Cambridge_21.pdf` → `pdf_cambridge_21`
- `document.pdf` → `pdf_document`

### What ChromaDB Stores
- **ID:** Unique chunk ID (`chunk_0`, `chunk_1`, ...)
- **Vector:** The 384-dimensional float array.
- **Document Text:** The actual textual excerpt.
- **Metadata:** Source page number (`{"page": 5}`).

---

## 9. Semantic Retrieval

When a question is asked:
1. The question is encoded using `all-MiniLM-L6-v2`.
2. ChromaDB performs an Approximate Nearest Neighbor (ANN) search using cosine distance.
3. The top-$K$ most relevant chunks are retrieved (default `top_k = 3`).

*Note on Distance Values:* A distance of `0.90` is a geometric vector metric—it does **not** mean 90% confidence or 90% accuracy. Lower distance indicates closer semantic proximity.

---

## 10. Gemini Generation & Grounding

The retrieved chunks are formatted into a strict grounding prompt:

```text
CONTEXT:
[Context 1 - Page 5]:
The Cambridge IELTS 21 Practice Tests provides authentic examination papers...

QUESTION:
Is this book for IELTS?
```

Gemini generates a response using **only** the supplied context. If the fact is not in the context, it states that the answer could not be found in the PDF.

---

## 11. Backend API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check verifying that the backend is operational. |
| `GET` | `/api/documents` | Lists all `.pdf` documents available in the `documents/` folder. |
| `POST` | `/api/documents/select` | Validates and indexes the requested PDF in ChromaDB. |
| `POST` | `/api/chat` | Performs vector search, prompts Gemini, and returns the answer with sources. |

### Example Request / Response

**Select Document:**
```bash
curl -X POST http://localhost:8000/api/documents/select \
  -H "Content-Type: application/json" \
  -d '{"filename": "Cambridge_21.pdf"}'
```
```json
{
  "name": "Cambridge_21.pdf",
  "pages": 144,
  "chunks": 218,
  "status": "ready"
}
```

**Chat Query:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"filename": "Cambridge_21.pdf", "question": "Is this book for IELTS?"}'
```
```json
{
  "answer": "Yes, this book provides authentic practice tests designed for the IELTS test.",
  "sources": [5, 10]
}
```

---

## 12. Frontend Flow & States

The Astro frontend (`frontend/src/pages/index.astro`) manages these interactive states:

1. **No Document Selected:** Prompts user to select a PDF. Input is disabled.
2. **Loading Documents:** Populates the dropdown from `GET /api/documents`.
3. **Processing Document:** Displays a status indicator while ChromaDB ingests and embeds the file.
4. **Document Ready:** Displays page count, chunk count, and enables the chat input.
5. **Thinking State:** Shows an animated thinking indicator while ChromaDB retrieves chunks and Gemini synthesizes an answer.
6. **Answer Received:** Renders the assistant message bubble along with subtle `Sources: Page X, Page Y` tags.
7. **Clear Chat:** Resets client-side chat messages without deleting ChromaDB vectors.
8. **Error Handling:** Displays friendly error banners if the backend is offline or an API fails.

---

## 13. Project Structure

```text
mini_rag/
│
├── backend/
│   └── main.py              # FastAPI application exposing health, documents, and chat endpoints
│
├── frontend/
│   ├── src/
│   │   └── pages/
│   │       └── index.astro  # Astro web UI (dark theme, vanilla TS/CSS)
│   ├── package.json         # Frontend package configuration
│   └── astro.config.mjs     # Astro server configuration (port 4321)
│
├── documents/
│   ├── Cambridge_21.pdf     # Sample IELTS test book (144 pages)
│   └── document.pdf         # Sample ML concepts document (3 pages)
│
├── chroma_db/               # Persistent ChromaDB vector database directory
├── .env                     # Secrets (GEMINI_API_KEY, GEMINI_MODEL)
├── .env.example             # Template for environment variables
├── requirements.txt         # Backend Python dependencies
├── .gitignore               # Ignores .venv/, .env, chroma_db/, node_modules/, dist/
└── README.md                # Comprehensive project documentation
```

---

## 14. Installation

### 1. Python Environment (Backend)
Python 3.10+ (Python 3.12 recommended) is required.

```bash
# From the mini_rag/ root directory
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Node.js Environment (Frontend)
Node 18+ (Node 22 recommended) is required.

```bash
cd frontend
npm install
cd ..
```

---

## 15. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Set your configuration:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> **Security Note:** `GEMINI_API_KEY` is loaded strictly by the FastAPI backend. It is **never** exposed to the Astro frontend or sent to the browser.

---

## 16. Running the Application

Open two terminal tabs:

### Terminal 1: Start the Backend (FastAPI)
```bash
source .venv/bin/activate
cd backend
uvicorn main:app --reload --port 8000
```
*Backend runs on: `http://localhost:8000`*

### Terminal 2: Start the Frontend (Astro)
```bash
cd frontend
npm run dev
```
*Frontend runs on: `http://localhost:4321`*

Open **http://localhost:4321** in your browser.

---

## 17. Example Usage Walkthrough

1. Open `http://localhost:4321` in your browser.
2. Select **`Cambridge_21.pdf`** from the dropdown and click **Load Document**.
3. Status banner updates to: `Cambridge_21.pdf — 144 pages • 218 chunks • Ready to chat`.
4. Ask: *"Is this book for IELTS?"*
5. Assistant responds: *"Yes, this book provides authentic practice tests designed for the IELTS test."* with `Sources: Page 5, Page 10`.
6. Ask: *"What is quantum computing?"*
7. Assistant responds: *"The provided PDF context does not contain information about quantum computing."*

---

## 18. Limitations

As an educational mini-project, several constraints exist:
- **Broad Summaries:** Localized chunk vector search excels at pinpoint questions (*"What is backpropagation?"*), but can struggle with whole-document summaries (*"Summarize the entire 200-page book"*).
- **Scanned PDFs:** Requires text-based PDFs. Scanned images without text layers require OCR.
- **Single Turn Focus:** The backend treats each query independently without conversational session memory.

---

## 19. Future Improvements

- Add hybrid keyword + dense vector search (BM25 + embeddings).
- Implement cross-encoder reranking (e.g., `bge-reranker`).
- Add conversational chat history buffering.
- Add multi-PDF cross-collection searching.
- Integrate OCR support for scanned documents.
