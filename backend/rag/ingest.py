"""
One-time (re-run when docs change) ingestion pipeline:
PDF -> page-aware chunks -> embeddings -> ChromaDB.

Run: python -m rag.ingest
"""
import os
import glob

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", "../data/medical_pdfs")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "medical_docs"

# Chunking tuned for dense clinical text: smaller chunks + overlap so a
# retrieved passage rarely gets cut mid-sentence or mid-dosage-table.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_pdf_pages(path: str) -> list[dict]:
    """Returns [{text, page_number, source}] — page number kept for citations."""
    reader = PdfReader(path)
    doc_name = os.path.basename(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page_number": i + 1, "source": doc_name})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({
                "text": piece,
                "page_number": page["page_number"],
                "source": page["source"],
            })
    return chunks


def build_index():
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}. Add WHO/CDC/textbook PDFs there first.")
        return

    all_chunks = []
    for path in pdf_paths:
        print(f"Reading {path} ...")
        pages = load_pdf_pages(path)
        all_chunks.extend(chunk_pages(pages))

    print(f"Total chunks: {len(all_chunks)}")

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Uses a local sentence-transformers model by default so ingestion doesn't
    # burn API quota — swap for a Gemini/OpenAI embedding function if preferred.
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"  # biomedical-tuned embeddings
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    documents = [c["text"] for c in all_chunks]
    metadatas = [{"source": c["source"], "page_number": c["page_number"]} for c in all_chunks]

    # Chroma has a batch-size ceiling — write in batches
    BATCH = 500
    for start in range(0, len(documents), BATCH):
        end = start + BATCH
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"Indexed {end if end < len(documents) else len(documents)}/{len(documents)}")

    print(f"Done. Persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    build_index()
