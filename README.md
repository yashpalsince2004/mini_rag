# Mini PDF RAG Chatbot

A deliberately small, educational **Retrieval-Augmented Generation (RAG)** system built from scratch in Python using **PyMuPDF**, **Sentence Transformers**, **ChromaDB**, and **Google Gemini**.

The primary purpose of this project is learning and understanding how the internal RAG pipeline works from first principles. Rather than hiding critical operations behind high-level frameworks like LangChain or LlamaIndex, this implementation exposes every phase of the pipeline directly: text parsing, sliding-window chunking, vector embedding, similarity search, prompt augmentation, and grounded LLM generation.

```text
PDF
 ↓
PyMuPDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Sentence Transformers
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Semantic Retrieval
 ↓
Retrieved Context
 ↓
Gemini
 ↓
Answer + Sources
```

---

## 1. What is RAG?

**Retrieval-Augmented Generation (RAG)** is an architecture that connects a Large Language Model (LLM) to external knowledge stores without retraining or fine-tuning the model.

### The Problem with Standard LLMs

In a standard LLM interaction, the model relies solely on parameters learned during its pretraining phase:

```text
User Question
    ↓
   LLM
    ↓
  Answer
```

This presents fundamental challenges:
- **Private Knowledge Gaps:** The LLM has never seen your private files, internal documentation, or newly published books.
- **Knowledge Cutoffs:** The model cannot answer questions about data created after its training period.
- **Hallucinations:** When faced with questions outside its training distribution, an LLM often invents plausible-sounding but factually incorrect answers.

### The RAG Solution

RAG bridges this gap by introducing a retrieval step before calling the LLM:

```text
User Question
    ↓
Retriever (ChromaDB)
    ↓
Relevant Document Chunks
    ↓
LLM + Retrieved Context (Prompt Augmentation)
    ↓
Grounded Answer
```

### The Three Major Stages of RAG

1. **Ingestion:** Documents are loaded, parsed into raw text, split into smaller manageable chunks, converted into numerical vectors (embeddings), and indexed in a vector store.
2. **Retrieval:** When a user asks a question, the query is converted into an embedding using the identical model. The vector database performs a geometric similarity search to retrieve the top-$K$ most relevant text chunks.
3. **Generation:** The retrieved chunks are injected alongside the user question into a structured prompt. The LLM generates a grounded answer strictly constrained to the provided context, citing source pages.

### Where RAG is Used
- **PDF Documents & Books:** Querying text books, syllabi, and multi-page guides.
- **Company Documents:** Searching internal policies, SOPs, and onboarding wikis.
- **Technical Manuals & Specs:** Extracting pinpoint configuration steps from 500-page appliance or software manuals.
- **Academic Research Papers:** Summarizing methodologies and factual findings across scientific literature.
- **Customer Knowledge Bases:** Powering support bots that cite help-desk articles accurately.

---

## 2. RAG Architecture in This Project

The architecture of this project cleanly separates **Document Ingestion** from **Query-Time Question Answering**.

```text
                 DOCUMENT INGESTION

PDF (e.g. Cambridge_21.pdf / document.pdf)
 │
 ▼
PyMuPDF (fitz)
 │
 ▼
Extracted Text (page-by-page)
 │
 ▼
Chunking (Sliding Window: 300 words, 50-word overlap)
 │
 ▼
Text Chunks (with page metadata)
 │
 ▼
Sentence Transformer (all-MiniLM-L6-v2)
 │
 ▼
Embeddings (384-dimensional dense vectors)
 │
 ▼
ChromaDB (Persistent local collection: pdf_documents)


                 QUERY TIME

User Question
 │
 ▼
Sentence Transformer (all-MiniLM-L6-v2)
 │
 ▼
Query Embedding
 │
 ▼
ChromaDB Similarity Search (Cosine distance calculation)
 │
 ▼
Top-K Relevant Chunks (e.g. K=3 or K=5)
 │
 ▼
Prompt Construction (CONTEXT + QUESTION + Grounding rules)
 │
 ▼
Gemini (gemini-2.5-flash)
 │
 ▼
Final Answer
 │
 ▼
Source Pages (deduplicated page citations)
```

**Key Architectural Distinction:**
- **Ingestion** occurs once when preparing the vector database (or when forced via `RESET_DB=true`).
- **Retrieval & Generation** execute dynamically on every query entered by the user in the interactive terminal loop.

---

## 3. Project Components

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Runtime** | Python (3.10+ / 3.12) | Core execution environment and pipeline orchestration. |
| **PDF Extraction** | PyMuPDF (`fitz`) | High-speed, page-by-page text extraction that retains source page numbers. |
| **Embedding Framework** | Sentence Transformers | Local deep learning framework that converts text strings into dense vector representations. |
| **Embedding Model** | `all-MiniLM-L6-v2` | Fast, lightweight 384-dimensional model optimized for semantic text similarity. |
| **Vector Database** | ChromaDB (`chromadb`) | Local persistent vector store that indexes chunk embeddings and performs nearest-neighbor vector search. |
| **Language Model** | Google Gemini API (`google-genai`) | LLM (`gemini-2.5-flash`) that synthesizes natural-language answers strictly grounded in retrieved chunks. |
| **Config Management** | `python-dotenv` | Loads API keys and runtime configuration from `.env` without hardcoding credentials. |
| **Environment Isolation** | `.venv` (Virtual Environment) | Isolates Python packages from global machine libraries to avoid dependency conflicts. |

---

## 4. PDF Ingestion Pipeline

Ingestion transforms unstructured binary PDF bytes into indexed, searchable vectors.

### Step 1 — Open the PDF
PyMuPDF (`fitz.open(pdf_path)`) opens the document file directly from disk without spawning external subprocesses or cloud parsers.

### Step 2 — Extract Text Page-by-Page
The function `extract_pdf_text()` iterates sequentially across all pages (`doc[page_index]`). It calls `.get_text().strip()` on each page. Pages containing no readable text (e.g., blank pages or raw scans without OCR) are skipped.

### Preserving Page Metadata
Each extracted unit retains its original 1-indexed page number:

```python
{
    "page": 6,
    "text": "Extracted text content from page 6..."
}
```

**Why Preserving Page Metadata is Essential:**
1. **Source Attribution:** Allows the LLM and the application to cite the exact page numbers backing an answer.
2. **Verification & Auditability:** Enables users to flip to the original document page to verify facts.
3. **Debugging:** If an answer is inaccurate, inspecting the retrieved page numbers immediately reveals whether the retriever pulled the wrong section.

---

## 5. Chunking

### Why Not Embed the Entire Document at Once?
1. **Context Limits:** Embedding models and LLMs have finite input windows.
2. **Information Dilution:** Embedding a 100-page book into a single vector compresses thousands of ideas into one point, washing out specific facts.
3. **Retrieval Granularity:** Similarity search works best when matching a specific question against a concise, focused passage.

### The Sliding-Window Word Chunking Configuration
The function `create_chunks()` partitions extracted text using a word-based sliding window:
- **`chunk_size = 300` words:** The target length of each discrete chunk.
- **`chunk_overlap = 50` words:** The number of words repeated from the end of the previous chunk.

```text
Chunk A: [Word 1  ..................  Word 250  Word 251 ........ Word 300]
                                                │
                          ◄── 50-word Overlap ──►
                                                │
Chunk B:                               [Word 251 ........ Word 300  Word 301 ........ Word 550]
```

### Why Overlap is Critical
Without overlap, an important sentence or definition split right at the 300-word boundary would be severed: half the premise in Chunk A, half the conclusion in Chunk B. Neither chunk alone would match a query about that topic. Overlap ensures semantic continuity across chunk borders.

### Chunk Size Tradeoffs

| Chunk Size | Advantages | Disadvantages |
| :--- | :--- | :--- |
| **Small Chunks** (e.g., 50–100 words) | Highly specific vector representation; minimal noise. | Lacks surrounding context; may sever complex thoughts. |
| **Large Chunks** (e.g., 500–1000 words) | Preserves rich broad context. | Dilutes specific facts; LLM prompt fills up quickly; higher vector storage footprint. |

Chunking strategy is one of the most consequential architectural decisions in RAG.

---

## 6. Embeddings

### Embeddings from First Principles
An embedding model converts human language into a list of floating-point numbers known as a **vector**. 

Conceptually:
```text
"Deep learning is a subset of machine learning."
                           ↓
[ 0.0214, -0.1820, 0.4431, -0.0092, ... 384 dimensions ... ]
```
*(Note: These numbers are illustrative).*

In this high-dimensional space, geometric distance reflects semantic meaning. Concepts that share similar meanings are mapped close together, even if they use entirely different words:

```text
"What is deep learning?"
       ~ is geographically close to ~
"Explain deep neural networks."
```

### The Embedding Model: `all-MiniLM-L6-v2`
This project utilizes the `all-MiniLM-L6-v2` Sentence Transformer:
- **Dimensions:** 384 floating-point coordinates per vector.
- **Speed:** Fast local inference on standard CPU hardware.
- **Training:** Fine-tuned on over 1 billion sentence pairs for semantic similarity.

### The Golden Rule of Vector Embeddings
> **The exact same embedding model must be used for both document chunks and user queries.**

```text
Document Chunk ──► [ all-MiniLM-L6-v2 ] ──► Vector A (in 384-D Space)
                                                    ▲
                                            Cosine Distance
                                                    ▼
User Question   ──► [ all-MiniLM-L6-v2 ] ──► Vector B (in 384-D Space)
```

Different models project text into entirely different vector spaces with incompatible coordinate systems. Comparing a vector from model X with a vector from model Y produces meaningless noise.

---

## 7. ChromaDB

ChromaDB serves as the local, persistent vector database.

### What ChromaDB Stores
For every chunk, ChromaDB maintains four synchronized data fields:
1. **ID:** Unique identifier string (e.g., `chunk_0`, `chunk_1`).
2. **Embedding:** The 384-dimensional vector float array.
3. **Document:** The raw chunk text string.
4. **Metadata:** Dictionary containing structured attributes (e.g., `{"page": 2}`).

```text
┌────────────────────────────────────────────────────────────────────────┐
│ ChromaDB Collection: "pdf_documents"                                   │
├──────────┬───────────────────────┬──────────────────┬──────────────────┤
│ ID       │ Vector (384 floats)   │ Document Text    │ Metadata         │
├──────────┼───────────────────────┼──────────────────┼──────────────────┤
│ chunk_0  │ [0.021, -0.182, ...]  │ "Artificial..."  │ {"page": 1}      │
│ chunk_1  │ [-0.014, 0.312, ...]  │ "Supervised..."  │ {"page": 1}      │
│ chunk_2  │ [0.104, -0.055, ...]  │ "Deep neural..." │ {"page": 2}      │
└──────────┴───────────────────────┴──────────────────┴──────────────────┘
```

### Local Persistence (`./chroma_db`)
ChromaDB is initialized with `chromadb.PersistentClient(path="./chroma_db")`. This saves the index and SQLite tables directly to disk. Once indexed, the PDF does not need to be parsed or embedded again on subsequent launches.

### Collection Management & Rebuilding with `RESET_DB=true`
The application registers its collection under `pdf_documents`. In `create_vector_database()`, the code stores the active chunk configuration in the collection's metadata:
```python
metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
```

The system automatically detects if `CHUNK_SIZE` or `CHUNK_OVERLAP` was changed in the code, or if `RESET_DB=true` is set in the environment:
```bash
RESET_DB=true python main.py
```

**When Rebuilding is Useful:**
- Changing chunk size or overlap.
- Swapping in a new PDF document.
- Changing the underlying embedding model.
- Clearing out stale or corrupted test vectors.

---

## 8. Vector Search & Retrieval

### The Retrieval Flow
When a user types a query into the terminal:

```text
User Question: "Where is deep learning discussed?"
       │
       ▼
Query Embedding (via all-MiniLM-L6-v2)
       │
       ▼
ChromaDB Vector Query (collection.query)
       │
       ▼
Cosine Distance Search across stored chunk vectors
       │
       ▼
Top-K Nearest Chunks returned (with document text, page metadata, distances)
```

### Semantic Search vs. Keyword Search
- **Keyword Search (e.g., grep, SQL LIKE):** Looks for exact lexical matches. If a user asks *"Where are deep neural networks covered?"* but the document only says *"deep multilayered artificial networks"*, keyword search can return zero results.
- **Semantic Vector Search:** Compares the mathematical orientation of vectors. It retrieves conceptually relevant passages even when zero words overlap.

### Top-$K$ Retrieval & Distances
The system queries ChromaDB for the top $K$ nearest vectors (configured via `TOP_K = 3` or `TOP_K = 5` in `main.py`).

**Example Terminal Output:**
```text
-------------------------------
Retrieved Context
-------------------------------

[1] Page 2 (Distance: 0.9038)
Deep Learning and Neural Networks Deep Learning is a specialized branch of Machine Learning founded on multilayered Artificial Neural Networks that utilize representation learning...

[2] Page 2 (Distance: 1.1824)
a learnable bias term, and pass the scalar result through an activation function. - Activation Functions: Mathematical operators such as ReLU...

[3] Page 1 (Distance: 1.3261)
Machine Learning and Artificial Intelligence Overview Artificial Intelligence (AI) refers to the computational simulation of human intelligence...
-------------------------------
```

### Understanding Distance Values
- Lower distance values indicate greater geometric similarity (closer vectors).
- **CRITICAL:** Distance values are **not** percentages. A distance of `0.90` does **not** mean "90% accuracy" or "90% similarity". It is the mathematical squared Euclidean or cosine distance in high-dimensional vector space.

---

## 9. Why Retrieval is the Heart of RAG

The final quality of a RAG application is bounded by the quality of its retrieval step.

```text
POOR RETRIEVAL                       EFFECTIVE RETRIEVAL
      │                                       │
      ▼                                       ▼
Irrelevant or noisy chunks            Accurate, targeted chunks
      │                                       │
      ▼                                       ▼
LLM receives misleading context       LLM receives direct evidence
      │                                       │
      ▼                                       ▼
Hallucination or failed answer        Grounded, accurate answer
```

> **Key Rule:** An advanced LLM (such as Gemini 2.5 Flash) cannot compensate for a poor retriever. If ChromaDB fails to pull the chunk containing the needed fact, the LLM will either hallucinate or state that the answer cannot be found.

---

## 10. Prompt Augmentation

Prompt augmentation is the exact moment where **Retrieval** connects to **Generation**. The retrieved text chunks are formatted into a single structured prompt:

```text
You are a helpful assistant answering questions based solely on the provided PDF context.

INSTRUCTIONS:
- Answer using the supplied PDF context.
- Do not invent information.
- If the answer is not present in the retrieved context, say that the answer could not be found in the PDF.
- Do not rely on outside knowledge when answering the question.
- Keep answers concise but useful.

CONTEXT:
[Context 1 - Page 2]:
Deep Learning is a specialized branch of Machine Learning...

[Context 2 - Page 2]:
Backpropagation computes the partial derivatives of an objective loss function...

[Context 3 - Page 1]:
Supervised Learning algorithms are trained on rich datasets of labeled examples...

QUESTION:
What is backpropagation?
```

This prompt is what gives RAG its name: the bare question is **augmented** with retrieved external context before being passed to the LLM.

---

## 11. Gemini Generation

Google Gemini (`gemini-2.5-flash`) serves as the synthesis engine.

### Division of Labor
- **ChromaDB's Role:** Information retrieval (*"Here are the most relevant paragraphs found in the PDF"*).
- **Gemini's Role:** Reading comprehension and natural language synthesis (*"Reading only these excerpts, here is a concise answer"*).

Gemini does **not** search the PDF file, parse the binary document, or calculate vector distances. It processes the prompt text provided by `generate_answer()`.

### Strict Grounding Rules
The system prompt explicitly commands Gemini:
1. Ground every claim in the provided `CONTEXT`.
2. Never invent facts or extrapolate beyond the text.
3. If the context does not contain the answer, explicitly declare that the answer could not be found in the PDF.

---

## 12. End-to-End Walkthrough

Consider an actual query session run against `Cambridge_21.pdf` (144 pages, 218 chunks, `CHUNK_SIZE = 300`, `CHUNK_OVERLAP = 50`):

```text
You: Is the book for any specific exam?
```

1. **Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")` encodes the query into a 384-dimensional vector.
2. **Search:** ChromaDB scans all 218 stored chunk vectors and identifies the closest matches by cosine distance.
3. **Retrieval:** Top chunks are retrieved from Page 5, Page 6, and Page 2.
4. **Augmentation:** The retrieved chunks are formatted into the `CONTEXT` block of the prompt.
5. **Generation:** Gemini inspects the context and generates:
   ```text
   Assistant:
   Yes, the book is specifically prepared for candidates preparing for the IELTS (International English Language Testing System) examination.

   Sources:
   - Page 2
   - Page 5
   - Page 6
   ```
6. **Result:** The user receives a direct answer backed by verified page citations.

---

## 13. Document-Level Questions and Limitations

### The Specific vs. Document-Level Challenge
During testing, an important limitation of simple RAG pipelines becomes visible:

- **Specific Question:** *"What exam is this book for?"* or *"What is backpropagation?"*
  - **Result:** **Success.** The answer exists in a concentrated sentence or paragraph that closely aligns with the question embedding.
- **Document-Level Question:** *"What is this entire book about?"* or *"Summarize this document."*
  - **Result:** **Difficult.** The overall theme of a 144-page book is not concentrated in any single 300-word chunk. ChromaDB returns arbitrary localized chunks that happen to contain general words.

This is a natural characteristic of chunk-level vector search, not a bug in the code.

### Architectural Approaches for Document-Level RAG (Future Exploration)
- **Document Summaries:** Storing a high-level summary chunk during ingestion.
- **Hierarchical Retrieval:** Searching parent summaries first, then drilling into child chunks.
- **Hybrid Search:** Combining dense vector retrieval with keyword ranking (BM25).
- **Reranking:** Passing top-20 retrieved chunks through a cross-encoder model to re-score relevance.

---

## 14. Hallucination & Out-of-Context Questions

What happens if a user asks a question completely unrelated to the PDF?

```text
You: What is quantum computing?
```

1. ChromaDB retrieves the closest chunks it has (e.g., general AI text from Page 1 and Page 2 with large distance scores like `1.52` and `1.61`).
2. Gemini evaluates the retrieved context against the prompt instructions.
3. Because quantum computing is nowhere in the context, Gemini obeys the grounding instruction:
   ```text
   Assistant:
   The answer could not be found in the PDF.

   Sources:
   - Page 1
   - Page 2
   ```

This prevents hallucinations and protects factual integrity in educational and enterprise applications.

---

## 15. Complete Code Flow

The application logic resides entirely within `main.py`. The execution flow is structured as follows:

```text
 1. load_dotenv()                               ──► Load GEMINI_API_KEY and GEMINI_MODEL
 2. os.path.exists(PDF_PATH)                    ──► Verify the PDF file exists
 3. extract_pdf_text(PDF_PATH)                  ──► PyMuPDF extracts text page-by-page
 4. create_chunks(pages)                        ──► Sliding window word-level chunking
 5. SentenceTransformer(EMBEDDING_MODEL_NAME)   ──► Load all-MiniLM-L6-v2 model
 6. create_vector_database(...)                 ──► Connect to ChromaDB (check config/rebuild)
 7. embedding_model.encode(texts)               ──► Generate 384-D vectors for chunks
 8. collection.add(...)                         ──► Insert IDs, text, vectors, metadata into ChromaDB
 9. main() REPL Loop                            ──► Display banner and prompt for user input
10. input("You: ")                              ──► Read user query (or handle 'exit')
11. embedding_model.encode([query])             ──► Generate query vector
12. retrieve(query, collection, ...)            ──► ChromaDB similarity search (Top-K)
13. Print Retrieved Chunks                      ──► Display chunks and cosine distances
14. generate_answer(query, chunks, ...)         ──► Construct prompt and call Gemini API
15. Print Assistant Answer                      ──► Output synthesized response
16. Print Sources                               ──► Output deduplicated source page numbers
```

---

## 16. Why We Don't Use LangChain

This project deliberately avoids LangChain, LlamaIndex, or other all-in-one frameworks for educational reasons:

```text
FRAMEWORK APPROACH (Hidden)          EDUCATIONAL APPROACH (This Project)
┌──────────────────────────┐        ┌───────────────────────────────────┐
│                          │        │ 1. PyMuPDF extracts text          │
│                          │        │ 2. Plain Python splits words      │
│  LangChain / LlamaIndex  │        │ 3. SentenceTransformer encodes    │
│  "Magic Black Box"       │  ──►   │ 4. ChromaDB inserts & searches    │
│                          │        │ 5. Python string formats prompt   │
│                          │        │ 6. Google GenAI SDK generates     │
└──────────────────────────┘        └───────────────────────────────────┘
```

When learning with high-level frameworks:
- Developers cannot see when or how embeddings are calculated.
- Vector distances and search mechanics are hidden behind abstract "retriever" classes.
- Prompt formatting and system instructions are concealed inside deep library templates.

By building RAG with raw underlying libraries, you understand exactly how the components interact. Once you understand this foundation, framework abstractions become trivial to understand and debug.

---

## 17. Installation

### 1. Create and Activate a Virtual Environment
Python 3.10+ (Python 3.12 recommended) is required.

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 18. Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Configure your API key and preferred model:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> **Security Note:** Never commit `.env` or real API keys to version control. The `.gitignore` file is configured to exclude `.env`.

---

## 19. Running the Project

### Standard Run
Ensure your target PDF (e.g., `Cambridge_21.pdf` or `document.pdf`) is in the project root:

```bash
source .venv/bin/activate
python main.py
```

### Rebuilding the Vector Database
To force ChromaDB to wipe and regenerate the collection:

```bash
RESET_DB=true python main.py
```

**Use `RESET_DB=true` when:**
- Modifying `CHUNK_SIZE` or `CHUNK_OVERLAP` in `main.py`.
- Switching to a different PDF file.
- Updating or replacing the PDF content.

---

## 20. Expected Terminal Flow

Below is an authentic execution transcript:

```text
$ RESET_DB=true python main.py
Processed 144 pages with readable text from Cambridge_21.pdf.
Loading embedding model (all-MiniLM-L6-v2)...
Rebuilding ChromaDB collection 'pdf_documents' (chunk_size: 300, overlap: 50)...
Creating embeddings...
Adding chunks to ChromaDB...

========================================
        Mini PDF RAG Chatbot
========================================

PDF: Cambridge_21.pdf
Pages: 144
Chunks: 218
Model: gemini-2.5-flash

RAG system ready.

Type your question.
Type 'exit' to quit.

You: on what page deep learning is

-------------------------------
Retrieved Context
-------------------------------

[1] Page 2 (Distance: 0.9038)
Deep Learning and Neural Networks Deep Learning is a specialized branch of Machine Learning founded on multilayered Artificial Neural Networks that utilize representation learning...

[2] Page 2 (Distance: 1.1824)
a learnable bias term, and pass the scalar result through an activation function...

[3] Page 1 (Distance: 1.3261)
Machine Learning and Artificial Intelligence Overview Artificial Intelligence (AI) refers to the computational simulation of human intelligence...
-------------------------------

Assistant:
Deep learning is discussed on page 2.

Sources:
- Page 1
- Page 2

You: exit
Goodbye!
```

---

## 21. Project Directory

```text
mini_rag/
│
├── main.py              # The complete RAG pipeline (extract, chunk, embed, store, retrieve, generate)
├── requirements.txt     # The 5 core packages: chromadb, pymupdf, sentence-transformers, google-genai, python-dotenv
├── .env.example         # Template for environment variables
├── .gitignore           # Ignores .venv/, .env, chroma_db/, and python bytecode
├── README.md            # In-depth architectural guide and educational documentation
│
├── Cambridge_21.pdf     # Primary test PDF (or document.pdf)
├── chroma_db/           # Local ChromaDB persistent database directory (auto-generated, git-ignored)
└── .venv/               # Isolated virtual environment (local, git-ignored)
```

---

## 22. RAG Terminology Cheat Sheet

| Term | Meaning |
| :--- | :--- |
| **RAG** | Retrieval-Augmented Generation: enhancing LLM prompts with external retrieved knowledge. |
| **Embedding** | A high-dimensional numerical vector representing the semantic meaning of text. |
| **Vector** | An array of numbers coordinates in mathematical space (e.g., 384 dimensions). |
| **Vector Database** | A specialized database designed to store, index, and query vectors by geometric distance. |
| **Chunk** | A discrete slice of text extracted from a larger document. |
| **Chunk Overlap** | Repeating a set number of words between adjacent chunks to preserve boundary context. |
| **Retriever** | The system component that searches an index and fetches relevant chunks for a query. |
| **Top-$K$** | The number ($K$) of closest matching chunks retrieved from the vector database. |
| **Similarity Search** | Finding vectors that are closest to a query vector in multi-dimensional space. |
| **Distance Metric** | The mathematical formula used to calculate proximity (e.g., Cosine or Euclidean distance). |
| **Metadata** | Associated structured information saved alongside a chunk (e.g., source page number). |
| **Context** | The retrieved text injected into an LLM prompt to ground the generated response. |
| **Prompt Augmentation** | Combining system instructions, retrieved context, and the user question into one prompt. |
| **Generation** | The LLM reading the augmented prompt and synthesizing a natural-language answer. |
| **Grounding** | Constraining an LLM to base its response strictly on supplied context. |
| **Hallucination** | When an LLM produces plausible but factually incorrect or unsupported claims. |

---

## 23. RAG vs. Normal LLM

```text
STANDARD LLM PIPELINE
User Question ──► [ LLM (Pretrained Weights Only) ] ──► Answer (May hallucinate private data)


RAG PIPELINE
User Question ──► [ Embedding ] ──► [ ChromaDB Vector Search ] ──► Retrieved Chunks
                                                                           │
                                                                           ▼
User Question + Retrieved Context ──► [ LLM ] ──► Grounded Answer + Source Citations
```

| Dimension | Standard LLM | RAG System |
| :--- | :--- | :--- |
| **Data Source** | Static training weights only | Live external documents + training weights |
| **Private Data Access** | None (unless retrained) | Immediate (via document ingestion) |
| **Update Speed** | Months (retraining/fine-tuning) | Seconds (re-indexing a new PDF) |
| **Citations** | Opaque (cannot verify source) | Precise (cites specific pages and chunks) |
| **Hallucinations** | Frequent on unknown topics | Heavily minimized via prompt grounding |

---

## 24. What This Project Teaches

- [x] **PDF Text Extraction:** Parsing page-by-page text streams using PyMuPDF.
- [x] **Sliding-Window Chunking:** Implementing custom word-count chunking with overlap.
- [x] **Vector Embeddings:** Transforming natural language into dense vectors using Sentence Transformers.
- [x] **Vector Database Mechanics:** Initializing persistent ChromaDB collections, adding documents, and writing metadata.
- [x] **Semantic Search:** Understanding approximate nearest neighbor search and interpreting vector distance metrics.
- [x] **Context Injection:** Structuring RAG prompts with distinct `CONTEXT` and `QUESTION` delimiters.
- [x] **LLM Grounding:** Prompt engineering rules that force strict factual adherence.
- [x] **Source Attribution:** Tracking and reporting exact PDF page numbers for citations.
- [x] **Graceful Error Handling:** Clearly separating retrieval success from downstream LLM API availability.

---

## 25. What This Project Does Not Implement

To keep the codebase readable, educational, and beginner-friendly, the following advanced features are intentionally excluded:
- Optical Character Recognition (OCR) for scanned images.
- Complex table extraction and layout parsing.
- Hybrid search (combining BM25 keyword matching with dense vectors).
- Cross-encoder reranking models.
- Multi-query expansion and query rewriting.
- Multi-document routing and metadata filtering.
- Conversational chat history / memory across turns.
- Streaming token output.
- Web UI, REST APIs, or microservice wrappers.

---

## 26. Future RAG Experiments

Once comfortable with this baseline pipeline, explore this step-by-step roadmap:

1. **Level 1 (Current Project):** Single PDF, word-based chunking, persistent ChromaDB, Gemini LLM.
2. **Level 2 (Chunk Tuning):** Experiment with chunk sizes (100 vs. 300 vs. 600 words) and measure retrieval precision.
3. **Level 3 (Top-$K$ Optimization):** Compare output quality between $K=2$, $K=5$, and $K=10$.
4. **Level 4 (Retrieval Evaluation):** Implement precision and recall metrics comparing retrieved chunks against labeled ground truth.
5. **Level 5 (Metadata Filtering):** Add chapter or section tags and use ChromaDB's `where` filter to scope queries.
6. **Level 6 (Hybrid Search):** Combine keyword scoring (BM25) with dense vector search to capture both exact terminology and semantics.
7. **Level 7 (Cross-Encoder Reranking):** Retrieve top-20 chunks and use a cross-encoder (e.g., `bge-reranker`) to re-score the top 3.
8. **Level 8 (Conversational Memory):** Add a sliding history buffer so users can ask follow-up questions (*"What did you mention on page 2?"*).
9. **Level 9 (Multi-PDF Ingestion):** Scale the ingestion pipeline to parse an entire directory of PDFs with file-level metadata.
10. **Level 10 (Production Architecture):** Implement asynchronous workers, streaming responses, and an evaluation suite (e.g., Ragas).

---

## 27. Important Design Principles

1. **Embedding Model Symmetry:** The exact model used to encode document chunks must encode the user question.
2. **Retrieval Dictates Answer Quality:** If the retriever fails to supply relevant chunks, the LLM cannot produce a grounded answer.
3. **More Context is Not Always Better:** Excessive chunks introduce noise, distract the LLM, and risk the "lost in the middle" phenomenon.
4. **Chunk Size Balances Precision and Context:** Smaller chunks pinpoint facts; larger chunks preserve narrative context.
5. **Overlap Prevents Information Loss:** Overlapping boundaries guarantee that concepts straddling two chunks remain coherent.
6. **Vector Distance is Not Percentage Similarity:** A distance of 0.90 is a geometric coordinate metric, not 90% confidence.
7. **Metadata Enables Verifiability:** Preserving page numbers transforms black-box LLM output into an auditable document citation system.
8. **Grounding Mitigates Hallucinations:** Explicit system instructions are required to prevent LLMs from substituting outside knowledge.
9. **RAG Does Not Guarantee Truth:** If the source PDF contains errors, the RAG system will faithfully reproduce those errors.
10. **Simplicity Precedes Abstraction:** Understanding the underlying math, storage, and prompt mechanics is essential before adopting high-level frameworks.

---

## 28. Final Mental Model

> **"RAG is not the LLM reading your entire PDF."**

An LLM does not read a 200-page book when you ask a question. Instead:
1. The PDF is sliced into indexable pieces.
2. Those pieces are mapped into geometric coordinates (vectors).
3. The vectors are stored in ChromaDB.
4. Your question is mapped into that same coordinate space.
5. ChromaDB finds the closest matching pieces.
6. Only those matching pieces are shown to the LLM.
7. The LLM reads only those snippets and answers your question.

```text
PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
ChromaDB
 ↓
        User Question
              ↓
          Embedding
              ↓
       Similarity Search
              ↓
       Relevant Chunks
              ↓
        Gemini + Context
              ↓
            Answer
```
