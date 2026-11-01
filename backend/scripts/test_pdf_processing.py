"""Run extraction and chunking checks against the local PDF document corpus."""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_pages
from app.rag.loader import load_pdf_pages

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

DOCUMENTS_DIR = BACKEND_DIR / "data" / "documents"


def validate_chunks(chunks: list[Document], source_pages: dict[str, dict[int, str]]) -> None:
    seen_ids: set[str] = set()

    for chunk in chunks:
        assert chunk.page_content.strip(), "Chunk text must not be empty"
        assert len(chunk.page_content) <= CHUNK_SIZE, "Chunk exceeds configured chunk size"
        assert {"source", "page", "chunk_id"}.issubset(chunk.metadata), "Required metadata is missing"

        source = chunk.metadata["source"]
        page = chunk.metadata["page"]
        chunk_id = chunk.metadata["chunk_id"]
        assert source in source_pages, f"Unknown source filename: {source}"
        assert isinstance(page, int) and page in source_pages[source], f"Invalid page for {source}: {page}"
        assert chunk.page_content in source_pages[source][page], f"Chunk text does not belong to {source}, page {page}"
        assert isinstance(chunk_id, str) and chunk_id and chunk_id not in seen_ids, "Chunk IDs must be unique"
        seen_ids.add(chunk_id)


def main() -> int:
    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda path: path.name.casefold())
    pages_by_source: dict[str, dict[int, str]] = defaultdict(dict)
    all_pages: list[Document] = []
    failed_pdfs: list[tuple[str, str]] = []
    warnings: list[str] = []
    processed_successfully = 0

    for pdf_path in pdf_paths:
        try:
            pages = load_pdf_pages(pdf_path)
        except Exception as error:
            failed_pdfs.append((pdf_path.name, f"{type(error).__name__}: {error}"))
            continue

        processed_successfully += 1
        if not pages:
            warnings.append(f"{pdf_path.name}: no extractable text; possibly scanned or image-only (OCR not run)")
        else:
            text_pages = {int(page.metadata["page"]): page.page_content for page in pages}
            pages_by_source[pdf_path.name].update(text_pages)
            all_pages.extend(pages)

    chunks = chunk_pages(all_pages)
    validate_chunks(chunks, pages_by_source)

    chunks_by_page: dict[tuple[str, int], list[Document]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_page[(chunk.metadata["source"], chunk.metadata["page"])].append(chunk)

    observed_overlaps: list[int] = []
    for page_chunks in chunks_by_page.values():
        for previous, following in zip(page_chunks, page_chunks[1:]):
            overlap = 0
            maximum = min(CHUNK_OVERLAP, len(previous.page_content), len(following.page_content))
            for size in range(1, maximum + 1):
                if previous.page_content[-size:] == following.page_content[:size]:
                    overlap = size
            assert overlap <= CHUNK_OVERLAP, "Adjacent chunks exceed the configured overlap"
            observed_overlaps.append(overlap)

    print("PDF processing report")
    print(f"PDFs discovered: {len(pdf_paths)}")
    print(f"PDFs processed successfully: {processed_successfully}")
    print(f"PDFs failed: {len(failed_pdfs)}")
    print(f"Pages containing extractable text: {len(all_pages)}")
    print(f"Total chunks generated: {len(chunks)}")
    print(f"Chunk settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}; pages chunked independently")
    if observed_overlaps:
        average_overlap = sum(observed_overlaps) / len(observed_overlaps)
        print(
            "Observed same-page overlap: "
            f"min={min(observed_overlaps)}, average={average_overlap:.1f}, max={max(observed_overlaps)} characters"
        )
    else:
        print("Observed same-page overlap: no adjacent chunks to compare")

    if warnings or failed_pdfs:
        print("Warnings and errors:")
        for warning in warnings:
            print(f"- WARNING: {warning}")
        for filename, error in failed_pdfs:
            print(f"- ERROR: {filename}: {error}")
    else:
        print("Warnings/errors: none")

    print("Validation: non-empty text, size limit, unique IDs, source filename, and page provenance passed")
    return 1 if failed_pdfs else 0


if __name__ == "__main__":
    raise SystemExit(main())