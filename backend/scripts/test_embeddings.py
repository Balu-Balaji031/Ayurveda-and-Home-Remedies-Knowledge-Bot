"""Generate and validate local embeddings for the current PDF corpus."""

from __future__ import annotations

import math
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.chunker import chunk_pages
from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_NAME,
    embed_documents,
)
from app.rag.loader import load_pdf_pages

DOCUMENTS_DIR = BACKEND_DIR / "data" / "documents"
BATCH_SIZE = 16


def main() -> int:
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda path: path.name.casefold())
    pages = []
    errors: list[str] = []

    for pdf_path in pdf_paths:
        try:
            pages.extend(load_pdf_pages(pdf_path))
        except Exception as error:
            errors.append(f"{pdf_path.name}: {type(error).__name__}: {error}")

    chunks = chunk_pages(pages)
    embedded_chunks = embed_documents(chunks, batch_size=BATCH_SIZE)

    assert len(embedded_chunks) == len(chunks), "Each non-empty input chunk must have one embedding"
    assert all(len(item.vector) == EMBEDDING_DIMENSIONS for item in embedded_chunks), "Embedding dimension mismatch"
    assert all(math.isfinite(value) for item in embedded_chunks for value in item.vector), "Embedding contains non-finite values"
    assert all(item.text == chunk.page_content for item, chunk in zip(embedded_chunks, chunks, strict=True)), "Text association changed"
    assert all(item.metadata == chunk.metadata for item, chunk in zip(embedded_chunks, chunks, strict=True)), "Metadata association changed"
    assert all({"source", "page", "chunk_id"}.issubset(item.metadata) for item in embedded_chunks), "Chunk metadata is incomplete"
    assert all(item.text.strip() for item in embedded_chunks), "Embedded text must not be empty"

    print("Embedding test report")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"PDFs discovered: {len(pdf_paths)}")
    print(f"Chunks received: {len(chunks)}")
    print(f"Embeddings generated: {len(embedded_chunks)}")
    print(f"Embedding dimensions: {EMBEDDING_DIMENSIONS}")
    print(f"All vectors have correct dimensions: yes ({len(embedded_chunks)} checked)")
    print(f"All vector values are finite: yes ({len(embedded_chunks)} checked)")
    print("Text and metadata associations preserved: yes")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Errors or warnings: {len(errors)}")
    for error in errors:
        print(f"- ERROR: {error}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())