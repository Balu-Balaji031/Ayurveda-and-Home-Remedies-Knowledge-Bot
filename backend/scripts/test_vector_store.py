"""Validate the real persistent ChromaDB corpus and repeat-ingestion behavior."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddedChunk
from app.rag.vector_store import (
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    get_collection_stats,
    ingest_chunks,
    initialize_vector_store,
    validate_embedded_chunks,
)
from scripts.ingest_documents import load_document_chunks


def validate_stored_chunks(collection, chunks) -> None:
    expected = {str(chunk.metadata["chunk_id"]): chunk for chunk in chunks}
    result = collection.get(
        ids=list(expected),
        include=["documents", "metadatas", "embeddings"],
    )
    assert len(result["ids"]) == len(expected), "Not all expected chunks are stored"
    assert len(set(result["ids"])) == len(result["ids"]), "Stored IDs are not unique"

    for index, record_id in enumerate(result["ids"]):
        source_chunk = expected[record_id]
        assert result["documents"][index] == source_chunk.page_content, f"Text changed for {record_id}"
        assert result["metadatas"][index] == source_chunk.metadata, f"Metadata changed for {record_id}"
        metadata = result["metadatas"][index]
        assert metadata["chunk_id"] == record_id
        assert metadata["source"] == source_chunk.metadata["source"]
        assert metadata["page"] == source_chunk.metadata["page"]
        embedding = result["embeddings"][index]
        assert len(embedding) == EMBEDDING_DIMENSIONS, f"Incorrect dimensions for {record_id}"
        assert all(math.isfinite(float(value)) for value in embedding), f"Non-finite embedding for {record_id}"


def test_reopened_process(expected_count: int, expected_ids: list[str]) -> str:
    code = r'''
import json, math, sys
sys.path.insert(0, sys.argv[1])
from app.rag.vector_store import initialize_vector_store, get_collection_stats
from app.rag.embeddings import EMBEDDING_DIMENSIONS
expected_count = int(sys.argv[2])
expected_ids = json.loads(sys.argv[3])
client, collection = initialize_vector_store()
stats = get_collection_stats(collection)
assert stats["count"] == expected_count, (stats["count"], expected_count)
result = collection.get(ids=expected_ids, include=["documents", "metadatas", "embeddings"])
assert len(result["ids"]) == len(expected_ids)
assert all(text and text.strip() for text in result["documents"])
assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in result["embeddings"])
assert all(math.isfinite(float(value)) for vector in result["embeddings"] for value in vector)
print(json.dumps({"count": stats["count"], "path": stats["persist_directory"], "records_reopened": len(result["ids"])}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(BACKEND_DIR), str(expected_count), __import__("json").dumps(expected_ids)],
        check=True,
        capture_output=True,
        text=True,
        cwd=BACKEND_DIR,
    )
    return completed.stdout.strip()


def test_invalid_records_rejected() -> int:
    valid_metadata = {"source": "test.pdf", "page": 1, "chunk_id": "test-1"}
    invalid_records = [
        EmbeddedChunk("  ", valid_metadata, [0.0] * EMBEDDING_DIMENSIONS),
        EmbeddedChunk("text", valid_metadata, [0.0] * (EMBEDDING_DIMENSIONS - 1)),
        EmbeddedChunk("text", valid_metadata, [math.nan] + [0.0] * (EMBEDDING_DIMENSIONS - 1)),
        EmbeddedChunk("text", {"source": "test.pdf", "page": 1}, [0.0] * EMBEDDING_DIMENSIONS),
    ]
    for record in invalid_records:
        try:
            validate_embedded_chunks([record])
        except ValueError:
            continue
        raise AssertionError("Invalid record was not rejected")
    return len(invalid_records)


def run_persistence_check() -> int:
    pdf_count, chunks, errors = load_document_chunks()
    assert not errors, f"PDF loading errors: {errors}"
    assert pdf_count == 3, f"Expected 3 PDFs, found {pdf_count}"
    expected_ids = [str(chunk.metadata["chunk_id"]) for chunk in chunks]
    assert len(expected_ids) == 299, f"Expected 299 chunks, found {len(expected_ids)}"

    client, collection = initialize_vector_store()
    initial_stats = get_collection_stats(collection)
    initial_count = int(initial_stats["count"])
    first_run = ingest_chunks(chunks, collection=collection, persist_directory=CHROMA_PERSIST_DIRECTORY)
    assert collection.count() >= initial_count + (first_run.inserted), "Collection count did not reflect inserted records"
    validate_stored_chunks(collection, chunks)
    after_first_run_count = collection.count()
    assert after_first_run_count == first_run.final_count

    print("ChromaDB vector-store test report")
    print(f"Collection initialization: PASS ({COLLECTION_NAME})")
    print(f"Database directory: {get_collection_stats(collection)['persist_directory']}")
    print(f"PDFs/chunks loaded: {pdf_count} PDFs / {len(chunks)} chunks")
    print(f"Records represented: PASS ({len(expected_ids)} expected chunk IDs)")
    print(f"Embedding dimensions and finite values: PASS ({len(expected_ids)} records, {EMBEDDING_DIMENSIONS} dimensions)")
    print("Text, metadata, source, page, and chunk ID: PASS")
    print(f"Invalid-record cases rejected: PASS ({test_invalid_records_rejected()} cases)")

    second_run = ingest_chunks(chunks, collection=collection, persist_directory=CHROMA_PERSIST_DIRECTORY)
    after_second_run_count = collection.count()
    assert after_second_run_count == after_first_run_count, "Record count changed on identical second ingestion"
    assert second_run.inserted == 0 and second_run.updated == 0, "Unchanged records should not be written again"
    assert second_run.unchanged == len(chunks), "The second run should reuse every stored record"
    print(f"Duplicate-ingestion prevention: PASS (count remains {after_second_run_count})")

    reopen_report = test_reopened_process(after_second_run_count, expected_ids)
    print(f"Separate-process persistence/reopen: PASS ({reopen_report})")
    print(f"Final collection record count: {after_second_run_count}")
    print("All vector-store tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_persistence_check())