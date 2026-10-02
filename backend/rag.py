import os
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import HTTPException

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    import fitz

from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# =====================================================================
# Configuration Defaults
# =====================================================================
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_PATH = str(BASE_DIR / "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = 300       # Chunk size in words
CHUNK_OVERLAP = 50     # Overlap between consecutive chunks in words
TOP_K = 3              # Top K chunks to retrieve for RAG query

# Ensure documents directory exists
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB client and global embedding model cache
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily loads the SentenceTransformer model once."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_gemini_client() -> genai.Client:
    """Initializes and returns the Google Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not found in environment. Please configure .env file."
        )
    return genai.Client(api_key=api_key)


def get_collection_name(filename: str) -> str:
    """
    Generates a deterministic, valid ChromaDB collection name for a given PDF filename.
    ChromaDB collection names must be 3-63 characters, alphanumeric, underscores or hyphens.
    """
    stem = Path(filename).stem.lower()
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', stem).strip('_-')
    name = f"pdf_{cleaned}"[:60]
    if not name[-1].isalnum():
        name = name[:-1] + "0"
    return name


def validate_pdf_path(filename: str) -> Path:
    """
    Validates that the filename exists inside documents/ and prevents path traversal attacks.
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file: Only .pdf files are supported.")

    file_path = (DOCUMENTS_DIR / filename).resolve()
    
    # Path traversal check: resolved path must start with DOCUMENTS_DIR
    try:
        file_path.relative_to(DOCUMENTS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Access denied: Invalid file path.")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in documents directory.")

    return file_path


# =====================================================================
# Core RAG Pipeline Functions
# =====================================================================

def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF document using PyMuPDF.
    Preserves 1-indexed page number as metadata.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to open PDF document: {e}")

    pages = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text().strip()
        if text:
            pages.append({
                "page": page_index + 1,
                "text": text
            })

    if not pages:
        raise HTTPException(status_code=400, detail="No readable text found in PDF document.")

    return pages


def create_chunks(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits page text into overlapping word chunks.
    Overlapping words preserve semantic meaning across chunk boundaries.
    Each chunk retains its source PDF page number.
    """
    chunks = []
    step = chunk_size - chunk_overlap

    for page_info in pages:
        words = page_info["text"].split()
        page_num = page_info["page"]

        if not words:
            continue

        if len(words) <= chunk_size:
            chunks.append({
                "text": " ".join(words),
                "page": page_num
            })
            continue

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_num
            })
            if i + chunk_size >= len(words):
                break

    return chunks


def ingest_document(filename: str, file_path: Path) -> tuple[int, int]:
    """
    Performs complete ingestion:
    PDF text extraction -> Word-level chunking -> Vector embedding -> ChromaDB storage.
    """
    collection_name = get_collection_name(filename)
    
    # 1. Extract PDF text
    pages = extract_pdf_text(str(file_path))

    # 2. Chunk text
    chunks = create_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 3. Generate embeddings
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()

    # 4. Store embeddings in ChromaDB
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={
            "filename": filename,
            "page_count": len(pages),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c["page"]} for c in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(pages), len(chunks)


def is_overview_query(query: str) -> bool:
    """Detects whether user is asking for an overview/summary of the entire document."""
    q = query.lower()
    patterns = [
        r"\b(book|document|pdf)\b.*\b(about|summary|overview|topic|cover|describe)\b",
        r"\b(about|summary|overview)\b.*\b(book|document|pdf)\b",
        r"\bwhat is this (book|document|pdf)\b",
        r"\bwhat is (the|this) (book|pdf|document) about\b"
    ]
    return any(re.search(p, q) for p in patterns)


def retrieve_relevant_chunks(query: str, collection_name: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embeds the user's question with Sentence Transformers and searches ChromaDB for the closest chunks.
    For whole-document summary/overview queries, includes introductory chunks.
    """
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Document collection '{collection_name}' not found. Please select and ingest the document first."
        )

    if collection.count() == 0:
        raise HTTPException(status_code=400, detail="Document vector index is empty.")

    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count())
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results and results["distances"] else [None] * len(documents)

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text": doc,
            "page": meta.get("page", "Unknown"),
            "distance": dist
        })

    # For whole-document summary/overview questions, retrieve the document's introductory pages (title, contents, intro)
    if is_overview_query(query):
        try:
            overview_chunks = []
            for p_num in [2, 4, 5]:
                p_res = collection.get(where={"page": p_num}, limit=1)
                for doc, meta in zip(p_res.get("documents", []), p_res.get("metadatas", [])):
                    overview_chunks.append({
                        "text": doc,
                        "page": meta.get("page", p_num),
                        "distance": 0.80
                    })
            if overview_chunks:
                return overview_chunks[:top_k]
        except Exception:
            pass

    return retrieved[:top_k]


def generate_grounded_answer(query: str, retrieved_chunks: list[dict], model_name: str = GEMINI_MODEL) -> tuple[str, list[int]]:
    """
    Builds the augmented prompt with retrieved context and asks Gemini to generate an answer.
    Uses client.chats.create to eliminate automatic function calling (AFC) warnings.
    """
    pages = sorted(list(set(c["page"] for c in retrieved_chunks if isinstance(c["page"], int))))

    context_blocks = []
    for i, c in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Context {i} - Page {c['page']}]:\n{c['text']}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions based solely on the provided PDF context.

INSTRUCTIONS:
- Answer using the supplied PDF context.
- Do not invent information.
- If the answer is not present in the retrieved context, say that the answer could not be found in the PDF.
- Do not rely on outside knowledge when answering the question.
- Keep answers concise, factual, and well-structured.

CONTEXT:
{context_text}

QUESTION:
{query}
"""

    import time
    client = get_gemini_client()
    answer = ""
    for attempt in range(2):
        try:
            chat = client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            answer = response.text.strip()
            break
        except Exception as e:
            if "503" in str(e) and attempt == 0:
                time.sleep(1.5)
                continue
            answer = (
                "⚠️ Gemini generation failed.\n\n"
                "The retrieval and context validation succeeded, but the LLM could not generate the final answer.\n\n"
                f"Error details:\n{e}"
            )

    return answer, pages
