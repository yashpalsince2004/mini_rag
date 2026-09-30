import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

try:
    from rag import (
        DOCUMENTS_DIR,
        TOP_K,
        GEMINI_MODEL,
        chroma_client,
        validate_pdf_path,
        get_collection_name,
        ingest_document,
        retrieve_relevant_chunks,
        generate_grounded_answer,
    )
    from jev_decisions import (
        classify_query_intent,
        evaluate_context_sufficiency,
        DEFAULT_JEV_MODEL,
    )
except ImportError:
    from .rag import (
        DOCUMENTS_DIR,
        TOP_K,
        GEMINI_MODEL,
        chroma_client,
        validate_pdf_path,
        get_collection_name,
        ingest_document,
        retrieve_relevant_chunks,
        generate_grounded_answer,
    )
    from .jev_decisions import (
        classify_query_intent,
        evaluate_context_sufficiency,
        DEFAULT_JEV_MODEL,
    )

# Initialize FastAPI App
app = FastAPI(
    title="Mini RAG API with Jev Decision Layer via OpenRouter",
    description="Educational RAG Backend powered by ChromaDB, TypeSafe Jev (OpenRouter), and Google Gemini"
)

# Configure CORS so Astro frontend (http://localhost:4321) can communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Request / Response Schemas
# =====================================================================

class DocumentSelectRequest(BaseModel):
    filename: str

class ChatRequest(BaseModel):
    filename: Optional[str] = None
    question: Optional[str] = None
    message: Optional[str] = None


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend service and configuration status."""
    has_openrouter = bool(
        os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENROUTER_API_KEY").strip().startswith("your_")
    )
    return {
        "status": "ok",
        "jev_model": os.getenv("JEV_MODEL", DEFAULT_JEV_MODEL),
        "openrouter_configured": has_openrouter,
        "gemini_model": GEMINI_MODEL,
        "documents_dir": str(DOCUMENTS_DIR)
    }


@app.get("/api/documents")
def list_documents():
    """
    Returns available PDF files from the local documents/ directory.
    Only allows .pdf files.
    """
    if not DOCUMENTS_DIR.exists():
        return {"documents": []}

    pdf_files = sorted([f.name for f in DOCUMENTS_DIR.iterdir() if f.is_file() and f.name.lower().endswith(".pdf")])
    return {
        "documents": [{"name": name} for name in pdf_files]
    }


@app.post("/api/documents/select")
def select_document(request: DocumentSelectRequest):
    """
    Validates selected PDF, checks if it is already indexed in ChromaDB,
    and ingests/indexes it if not already ready.
    """
    file_path = validate_pdf_path(request.filename)
    collection_name = get_collection_name(request.filename)

    # Check if already indexed in ChromaDB
    try:
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        if count > 0:
            metadata = collection.metadata or {}
            stored_pages = metadata.get("page_count", 0)
            return {
                "name": request.filename,
                "pages": stored_pages,
                "chunks": count,
                "status": "ready"
            }
    except Exception:
        pass

    # If not indexed, run ingestion
    pages_count, chunks_count = ingest_document(request.filename, file_path)

    return {
        "name": request.filename,
        "pages": pages_count,
        "chunks": chunks_count,
        "status": "ready"
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    Executes the Jev-orchestrated RAG pipeline via OpenRouter:
    1. Jev classifies query intent (Decision 1) & gates retrieval (Decision 2).
    2. Non-document queries return immediately (no retrieval, no generation).
    3. ChromaDB retrieves top-K chunks for document questions.
    4. Jev evaluates context quality & sufficiency (Decision 3 & 4).
    5. Gemini generates answer only if context is sufficient.
    """
    raw_query = request.question or request.message or ""
    question = raw_query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    target_doc = request.filename or "Cambridge_21.pdf"
    validate_pdf_path(target_doc)
    collection_name = get_collection_name(target_doc)
    debug_mode = os.getenv("DEBUG_DECISIONS", "true").lower() in ("true", "1", "yes")

    print("\n" + "=" * 50)
    print(f"Incoming Query: '{question}'")
    print(f"Document: {target_doc}")
    print("=" * 50)

    # -------------------------------------------------------------
    # 1. Jev Decision 1 & 2: Query Routing and Retrieval Gating
    # -------------------------------------------------------------
    routing = classify_query_intent(question, filename=target_doc)

    if routing.query_type == "greeting":
        decision_data = {
            "route": routing.query_type,
            "retrieval_used": False,
            "context_sufficient": False,
            "query_type": routing.query_type,
            "should_retrieve": False,
            "should_generate": False,
            "context_quality": "n/a",
            "confidence": routing.confidence,
            "engine": routing.engine,
            "is_fallback": routing.is_fallback
        }
        res = {
            "answer": "Hello! Ask me anything about the selected PDF.",
            "sources": []
        }
        if debug_mode:
            res["decision"] = decision_data
        return res

    if routing.query_type == "help":
        decision_data = {
            "route": routing.query_type,
            "retrieval_used": False,
            "context_sufficient": False,
            "query_type": routing.query_type,
            "should_retrieve": False,
            "should_generate": False,
            "context_quality": "n/a",
            "confidence": routing.confidence,
            "engine": routing.engine,
            "is_fallback": routing.is_fallback
        }
        res = {
            "answer": (
                "I am an educational RAG assistant.\n\n"
                "1. Select a PDF from the dropdown above.\n"
                "2. Click **Load Document** to index chunks in ChromaDB.\n"
                "3. Ask any question about its contents.\n\n"
                "TypeSafe Jev evaluates your query via OpenRouter to prevent unnecessary retrieval, "
                "verifies retrieved context, and Google Gemini generates verified answers."
            ),
            "sources": []
        }
        if debug_mode:
            res["decision"] = decision_data
        return res

    if routing.query_type == "unsupported":
        decision_data = {
            "route": routing.query_type,
            "retrieval_used": False,
            "context_sufficient": False,
            "query_type": routing.query_type,
            "should_retrieve": False,
            "should_generate": False,
            "context_quality": "n/a",
            "confidence": routing.confidence,
            "engine": routing.engine,
            "is_fallback": routing.is_fallback
        }
        res = {
            "answer": "I can help you with questions about the selected PDF.",
            "sources": []
        }
        if debug_mode:
            res["decision"] = decision_data
        return res

    if routing.query_type == "clarification_needed":
        decision_data = {
            "route": routing.query_type,
            "retrieval_used": False,
            "context_sufficient": False,
            "query_type": routing.query_type,
            "should_retrieve": False,
            "should_generate": False,
            "context_quality": "n/a",
            "confidence": routing.confidence,
            "engine": routing.engine,
            "is_fallback": routing.is_fallback
        }
        res = {
            "answer": "Could you clarify what you'd like to know about the selected document?",
            "sources": []
        }
        if debug_mode:
            res["decision"] = decision_data
        return res

    # -------------------------------------------------------------
    # 2. ChromaDB Retrieval (Executed only when should_retrieve is True)
    # -------------------------------------------------------------
    print(f"\n[RAG] Executing vector similarity search in ChromaDB ({collection_name})...")
    retrieved_chunks = retrieve_relevant_chunks(question, collection_name, top_k=TOP_K)
    print(f"[RAG] Retrieved {len(retrieved_chunks)} chunks.")
    print("[RAG] Top results:")
    for i, c in enumerate(retrieved_chunks, start=1):
        dist_str = f"distance {c['distance']:.4f}" if c.get("distance") is not None else "distance N/A"
        print(f"  {i}. Page {c['page']} — {dist_str}")

    # -------------------------------------------------------------
    # 3. Jev Decision 3 & 4: Retrieval Quality and Generation Gating
    # -------------------------------------------------------------
    context_decision = evaluate_context_sufficiency(question, retrieved_chunks, filename=target_doc)

    # Top result metadata for inspection
    top_page = retrieved_chunks[0]["page"] if retrieved_chunks else None
    top_dist = round(retrieved_chunks[0]["distance"], 4) if retrieved_chunks and retrieved_chunks[0].get("distance") is not None else None

    if not context_decision.should_generate:
        print("[GEMINI] Generation skipped: Context is insufficient or unverified.")
        decision_data = {
            "route": routing.query_type,
            "retrieval_used": True,
            "context_sufficient": False,
            "query_type": routing.query_type,
            "should_retrieve": True,
            "should_generate": False,
            "context_quality": context_decision.context_quality,
            "confidence": context_decision.confidence,
            "noul_sufficiency": context_decision.noul_sufficiency,
            "engine": context_decision.engine,
            "is_fallback": context_decision.is_fallback,
            "retrieved_count": len(retrieved_chunks),
            "top_page": top_page,
            "top_distance": top_dist
        }
        res = {
            "answer": "I couldn't find enough information about that in the selected PDF.",
            "sources": []
        }
        if debug_mode:
            res["decision"] = decision_data
        return res

    # -------------------------------------------------------------
    # 4. Gemini Answer Generation (Executed only when context is sufficient)
    # -------------------------------------------------------------
    print(f"\n[GEMINI] Generating answer with {GEMINI_MODEL}...")
    answer, sources = generate_grounded_answer(question, retrieved_chunks)

    decision_data = {
        "route": routing.query_type,
        "retrieval_used": True,
        "context_sufficient": True,
        "query_type": routing.query_type,
        "should_retrieve": True,
        "should_generate": True,
        "context_quality": context_decision.context_quality,
        "confidence": context_decision.confidence,
        "noul_sufficiency": context_decision.noul_sufficiency,
        "engine": context_decision.engine,
        "is_fallback": context_decision.is_fallback,
        "retrieved_count": len(retrieved_chunks),
        "top_page": top_page,
        "top_distance": top_dist
    }
    res = {
        "answer": answer,
        "sources": sources
    }
    if debug_mode:
        res["decision"] = decision_data
    return res
