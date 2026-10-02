import os
import sys
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
try:
    # pyrefly: ignore [missing-import]
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    # pyrefly: ignore [missing-import]
    import fitz
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import chromadb
# pyrefly: ignore [missing-import]
from google import genai

# Load environment variables early for configuration
load_dotenv()

# =====================================================================
# Configuration
# =====================================================================
DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")
DEFAULT_PDF = os.path.join(DOCS_DIR, "Cambridge_21.pdf") if os.path.exists(os.path.join(DOCS_DIR, "Cambridge_21.pdf")) else (
    os.path.join(DOCS_DIR, "document.pdf") if os.path.exists(os.path.join(DOCS_DIR, "document.pdf")) else "Cambridge_21.pdf"
)
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CHUNK_SIZE = 300       # Chunk size in words (fine-grained chunks)
CHUNK_OVERLAP = 50     # Overlap between consecutive chunks in words
TOP_K = 5              # Number of relevant chunks to retrieve


# =====================================================================
# Step 1 & 2: PDF Loading and Text Extraction
# =====================================================================
def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extracts text page-by-page from a PDF document using PyMuPDF.
    Preserves the original 1-indexed page number as metadata.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text().strip()
        
        # Skip empty pages
        if not text:
            continue

        pages.append({
            "page": page_index + 1,
            "text": text
        })

    print(f"Processed {len(pages)} pages with readable text from {pdf_path}.")
    return pages


# =====================================================================
# Step 3: Text Chunking
# =====================================================================
def create_chunks(pages: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Splits page text into overlapping word chunks.
    Overlapping ensures context is not lost across chunk boundaries.
    Each chunk retains its source PDF page number.
    """
    chunks = []
    step = chunk_size - chunk_overlap

    for page_info in pages:
        words = page_info["text"].split()
        page_num = page_info["page"]

        if not words:
            continue

        # If page has fewer or equal words than chunk_size, store as a single chunk
        if len(words) <= chunk_size:
            chunks.append({
                "text": " ".join(words),
                "page": page_num
            })
            continue

        # Create overlapping chunks
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_num
            })
            if i + chunk_size >= len(words):
                break

    return chunks


# =====================================================================
# Step 4 & 5: Embedding Generation and ChromaDB Storage
# =====================================================================
def create_vector_database(
    chunks: list[dict],
    embedding_model: SentenceTransformer,
    db_path: str = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    reset: bool = False
):
    """
    Initializes a persistent ChromaDB vector store and populates it
    with chunk embeddings, raw text, and page metadata.
    Automatically detects if chunking configuration changed and rebuilds
    the collection to prevent reusing stale or incompatible chunks.
    """
    api_key = os.getenv("CHROMA_API_KEY")
    if api_key and not api_key.strip().startswith("your_"):
        tenant = os.getenv("CHROMA_TENANT", "default_tenant")
        database = os.getenv("CHROMA_DATABASE", "mini_rag")
        print(f"Connecting to hosted Chroma Cloud (tenant: {tenant}, database: {database})...")
        try:
            client = chromadb.CloudClient(
                api_key=api_key.strip(),
                tenant=tenant.strip(),
                database=database.strip()
            )
            print("Successfully connected to Chroma Cloud!")
        except Exception as e:
            print(f"Warning: Failed to connect to Chroma Cloud ({e}). Using local storage.")
            client = chromadb.PersistentClient(path=db_path)
    else:
        client = chromadb.PersistentClient(path=db_path)
    
    # Check if collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    
    if collection_name in existing_collections:
        collection = client.get_collection(name=collection_name)
        metadata = collection.metadata or {}
        stored_chunk_size = metadata.get("chunk_size")
        stored_chunk_overlap = metadata.get("chunk_overlap")

        # Detect if configuration changed or explicit reset requested
        config_mismatch = (stored_chunk_size != chunk_size or stored_chunk_overlap != chunk_overlap)
        env_reset = os.getenv("RESET_DB", "false").lower() in ("true", "1", "yes")

        if reset or env_reset or config_mismatch:
            print(f"Rebuilding ChromaDB collection '{collection_name}' (chunk_size: {chunk_size}, overlap: {chunk_overlap})...")
            client.delete_collection(name=collection_name)
            collection = client.create_collection(
                name=collection_name,
                metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            )
        elif collection.count() > 0:
            print(f"ChromaDB already contains {collection.count()} chunks.")
            print("Using existing vector database.")
            return collection
    else:
        collection = client.create_collection(
            name=collection_name,
            metadata={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
        )

    print("Creating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts).tolist()

    print("Adding chunks to ChromaDB...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"page": chunk["page"]} for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    return collection


# =====================================================================
# Step 6 & 7: Question Embedding and Similarity Search (Stage 1: Retrieval)
# =====================================================================
def is_overview_query(query: str) -> bool:  
    """Detects whether user is asking for an overview/summary of the entire document."""
    import re   
    q = query.lower()
    patterns = [
        r"\b(book|document|pdf)\b.*\b(about|summary|overview|topic|cover|describe)\b",
        r"\b(about|summary|overview)\b.*\b(book|document|pdf)\b",
        r"\bwhat is this (book|document|pdf)\b",
        r"\bwhat is (the|this) (book|pdf|document) about\b"
    ]
    return any(re.search(p, q) for p in patterns)


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K
) -> list[dict]:
    """
    Stage 1: Retrieval Pipeline
    Question -> Embedding -> ChromaDB -> Retrieved chunks
    
    Embeds the user's question using the same embedding model,
    searches ChromaDB for the closest vectors, and returns top chunks with distances.
    For whole-document overview queries, injects introductory chunks.
    """
    # Embed question with the same model used for chunks
    query_embedding = embedding_model.encode([query]).tolist()

    # Query ChromaDB for top_k most similar chunks (including distances)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    retrieved_chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results and results["distances"] else [None] * len(documents)

    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved_chunks.append({
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
                retrieved_chunks = overview_chunks[:top_k]
        except Exception:
            pass

    # Educational output: display retrieved context and distances
    print("\n-------------------------------")
    print("Retrieved Context")
    print("-------------------------------")
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        dist_info = f" (Distance: {chunk['distance']:.4f})" if chunk["distance"] is not None else ""
        print(f"\n[{idx}] Page {chunk['page']}{dist_info}")
        print(chunk["text"])
    print("\n-------------------------------")

    return retrieved_chunks


# =====================================================================
# Step 8, 9 & 10: Prompt Construction & LLM Generation (Stage 2: Generation)
# =====================================================================
def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    gemini_client: genai.Client,
    model_name: str = GEMINI_MODEL
) -> tuple[str, list[int]]:
    """
    Stage 2: Generation Pipeline
    Question + Retrieved chunks -> Gemini -> Answer
    
    Constructs a RAG prompt combining retrieved context and user question,
    calls Google Gemini API with the configured model, and returns the generated answer with sources.
    Clearly distinguishes Gemini generation failures from retrieval failures.
    """
    # Extract unique page numbers as sources
    pages = sorted(list(set(chunk["page"] for chunk in retrieved_chunks if isinstance(chunk["page"], int))))

    # Format retrieved context blocks
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[Context {i} - Page {chunk['page']}]:\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks)

    # RAG prompt with clear separation of context and question
    prompt = f"""You are a helpful assistant answering questions based solely on the provided PDF context.

INSTRUCTIONS:
- Answer using the supplied PDF context.
- Do not invent information.
- If the answer is not present in the retrieved context, say that the answer could not be found in the PDF.
- Do not rely on outside knowledge when answering the question.
- Keep answers concise but useful.

CONTEXT:
{context_text}

QUESTION:
{query}
"""

    # Generate answer with Gemini (using chat session to eliminate AFC warning and retrying temporary 503 spikes)
    import time
    answer = ""
    for attempt in range(2):
        try:
            chat = gemini_client.chats.create(model=model_name)
            response = chat.send_message(prompt)
            answer = response.text.strip()
            break
        except Exception as e:
            if "503" in str(e) and attempt == 0:
                time.sleep(1.5)
                continue
            answer = (
                "⚠️ Gemini generation failed.\n\n"
                "The retrieval step succeeded, but the LLM could not\n"
                "generate the final answer.\n\n"
                f"Error:\n{e}"
            )

    return answer, pages


# =====================================================================
# Main Application Flow
# =====================================================================
def main():
    # Load environment variables
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Error handling: Missing API key
    if not gemini_api_key:
        print("GEMINI_API_KEY not found.")
        print("Please create a .env file and add your Gemini API key.")
        sys.exit(1)

    # Error handling: Missing PDF
    if not os.path.exists(PDF_PATH):
        print(f"{PDF_PATH} not found.")
        print(f"Please place a PDF named {PDF_PATH} in the project directory.")
        sys.exit(1)

    # Initialize Gemini client
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        sys.exit(1)

    # Step 1 & 2: Load and extract text from PDF
    pages = extract_pdf_text(PDF_PATH)
    if not pages:
        print("No readable text was found in the PDF.")
        sys.exit(1)

    # Step 3: Chunk text with fine-grained configuration
    chunks = create_chunks(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Step 4: Load embedding model
    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Step 5: Store in ChromaDB vector database (with auto-rebuild on config change)
    collection = create_vector_database(
        chunks,
        embedding_model,
        db_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    # CLI Ready Banner
    print("\n========================================")
    print("        Mini PDF RAG Chatbot")
    print("========================================")
    print(f"\nPDF: {PDF_PATH}")
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model: {gemini_model}")
    print("\nRAG system ready.")
    print("\nType your question.")
    print("Type 'exit' to quit.")

    # Interactive Question-Answering Loop
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Jev Decision 1 & 2: Query Routing & Retrieval Gating
        try:
            from backend.jev_decisions import classify_query_intent, evaluate_context_sufficiency
            routing = classify_query_intent(query, filename=os.path.basename(PDF_PATH))
        except Exception:
            routing = None

        if routing and routing.query_type == "greeting":
            print("\nAssistant:")
            print("Hello! Ask me anything about the loaded PDF.")
            continue

        if routing and routing.query_type == "help":
            print("\nAssistant:")
            print("Type any question about the contents of the document. I will retrieve verified facts and answer.")
            continue

        if routing and routing.query_type == "unsupported":
            print("\nAssistant:")
            print("I can help you with questions about the selected PDF.")
            continue

        # Stage 1: Retrieval (Question -> Embedding -> ChromaDB -> Retrieved chunks)
        print(f"\n[RAG] Searching ChromaDB for: '{query}'")
        retrieved_chunks = retrieve(query, collection, embedding_model, top_k=TOP_K)
        print(f"[RAG] Retrieved {len(retrieved_chunks)} chunks.")

        # Jev Decision 3 & 4: Retrieval Quality & Generation Gating
        if routing:
            chunks_data = [
                {"text": c.get("text", ""), "page": c.get("page", 0), "distance": c.get("distance", None)}
                for c in retrieved_chunks
            ]
            context_decision = evaluate_context_sufficiency(query, chunks_data, filename=os.path.basename(PDF_PATH))
            if not context_decision.should_generate:
                print("\nAssistant:")
                print("I couldn't find enough information about that in the selected PDF.")
                continue

        # Stage 2: Generation (Question + Retrieved chunks -> Gemini -> Answer)
        print(f"\n[GEMINI] Generating answer with {gemini_model}...")
        answer, sources = generate_answer(query, retrieved_chunks, gemini_client, model_name=gemini_model)

        print("\nAssistant:")
        print(answer)

        if sources:
            print("\nSources:")
            for page in sources:
                print(f"- Page {page}")


if __name__ == "__main__":
    main()
