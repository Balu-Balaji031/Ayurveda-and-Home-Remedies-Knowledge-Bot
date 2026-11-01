"""Ingest the existing PDF corpus into persistent AyurBot ChromaDB storage."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from langchain_core.documents import Document

from app.rag.chunker import chunk_pages
from app.rag.loader import load_pdf_pages
from app.rag.vector_store import (
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    get_collection_stats,
    ingest_chunks,
    initialize_vector_store,
)

DOCUMENTS_DIR = BACKEND_DIR / "data" / "documents"
EMBEDDING_BATCH_SIZE = 16


def load_document_chunks() -> tuple[int, list[Document], list[str]]:
    """Use the existing loader and chunker; return discovered files, chunks, errors."""
    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(f"PDF documents directory does not exist: {DOCUMENTS_DIR}")

    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda path: path.name.casefold())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF documents were found in {DOCUMENTS_DIR}")

    pages: list[Document] = []
    errors: list[str] = []
    for pdf_path in pdf_paths:
        try:
            pages.extend(load_pdf_pages(pdf_path))
        except Exception as error:
            errors.append(f"{pdf_path.name}: {type(error).__name__}: {error}")

    if not pages:
        raise RuntimeError("No extractable PDF pages were available for chunking")
    return len(pdf_paths), chunk_pages(pages), errors


def main() -> int:
    try:
        pdf_count, chunks, load_errors = load_document_chunks()
        client, collection = initialize_vector_store()
        result = ingest_chunks(
            chunks,
            collection=collection,
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_batch_size=EMBEDDING_BATCH_SIZE,
        )
        stats = get_collection_stats(collection)
    except Exception as error:
        print(f"Ingestion failed: {type(error).__name__}: {error}")
        return 1

    print("AyurBot document ingestion")
    print(f"PDFs discovered: {pdf_count}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Embeddings generated: {result.embeddings_generated}")
    print(f"Unchanged embeddings reused: {result.unchanged}")
    print(f"Records inserted: {result.inserted}")
    print(f"Records updated: {result.updated}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Final collection count: {stats['count']}")
    print(f"Persistent directory: {stats['persist_directory']}")
    print(f"Errors: {len(load_errors)}")
    for load_error in load_errors:
        print(f"- {load_error}")

    return 1 if load_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())