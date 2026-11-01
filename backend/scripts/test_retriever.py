"""Test real Chroma retrieval and isolated retriever failure cases."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag import retriever
from app.rag.embeddings import EMBEDDING_DIMENSIONS, embed_texts
from app.rag.retriever import (
    DEFAULT_RETRIEVAL_COUNT,
    KnowledgeBaseUnavailableError,
    QueryEmbeddingError,
    RetrievalError,
    RetrievalResult,
    retrieve_documents,
)
from app.rag.vector_store import initialize_vector_store


class FakeCollection:
    def __init__(self, count=1, result=None, error=None):
        self._count = count
        self._result = result or {
            "ids": [["sample-p0001-c0001"]],
            "documents": [["A source passage." ]],
            "metadatas": [[{"source": "sample.pdf", "page": 1, "chunk_id": "sample-p0001-c0001"}]],
            "distances": [[0.25]],
        }
        self._error = error
        self.query_args = None

    def count(self):
        return self._count

    def query(self, **kwargs):
        self.query_args = kwargs
        if self._error:
            raise self._error
        return self._result


def run_isolated_tests() -> int:
    checks = 0

    fake = FakeCollection()
    with patch.object(retriever, "embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]) as embed_mock:
        records = retrieve_documents("a valid question", collection=fake)
    assert embed_mock.call_args.args[0] == ["a valid question"]
    assert len(fake.query_args["query_embeddings"][0]) == EMBEDDING_DIMENSIONS
    assert fake.query_args["n_results"] == min(DEFAULT_RETRIEVAL_COUNT, fake.count())
    assert records == [RetrievalResult("A source passage.", "sample.pdf", 1, "sample-p0001-c0001", 0.25, 0.75)]
    checks += 1

    with patch.object(retriever, "embed_texts") as embed_mock:
        try:
            retrieve_documents("  ", collection=FakeCollection())
        except ValueError:
            assert not embed_mock.called
            checks += 1
        else:
            raise AssertionError("Empty question was not rejected")

    with patch.object(retriever, "embed_texts", return_value=[[0.0] * (EMBEDDING_DIMENSIONS - 1)]):
        try:
            retrieve_documents("question", collection=FakeCollection())
        except QueryEmbeddingError:
            checks += 1
        else:
            raise AssertionError("Invalid vector dimensions were not rejected")

    with patch.object(retriever, "embed_texts", return_value=[[math.nan] + [0.0] * (EMBEDDING_DIMENSIONS - 1)]):
        try:
            retrieve_documents("question", collection=FakeCollection())
        except QueryEmbeddingError:
            checks += 1
        else:
            raise AssertionError("Non-finite query vector was not rejected")

    with patch.object(retriever, "embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]):
        assert retrieve_documents("question", collection=FakeCollection(count=0)) == []
        checks += 1

    with patch.object(retriever, "embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]):
        with patch.object(retriever, "_open_existing_collection", side_effect=KnowledgeBaseUnavailableError("missing collection")):
            try:
                retrieve_documents("question")
            except KnowledgeBaseUnavailableError:
                checks += 1
            else:
                raise AssertionError("Missing collection error was not propagated")

    with patch.object(retriever, "embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]):
        try:
            retrieve_documents("question", retrieval_count=0, collection=FakeCollection())
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Invalid retrieval count was not rejected")

    with patch.object(retriever, "embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]):
        try:
            retrieve_documents("question", collection=FakeCollection(error=RuntimeError("internal detail")))
        except RetrievalError as error:
            assert "internal detail" not in str(error)
            checks += 1
        else:
            raise AssertionError("Chroma query failure was not safely wrapped")

    return checks


def run_real_chroma_test() -> int:
    _, collection = initialize_vector_store()
    count = collection.count()
    assert count > 0, "Existing Chroma knowledge base is empty"
    records = retrieve_documents(
        "What do the documents say about Ayurvedic home remedies?",
        retrieval_count=DEFAULT_RETRIEVAL_COUNT,
        collection=collection,
    )
    assert records, "Real Chroma retrieval returned no results"
    assert len(records) <= DEFAULT_RETRIEVAL_COUNT
    for record in records:
        assert record.text.strip()
        assert Path(record.source).name == record.source
        assert record.page >= 1
        assert record.chunk_id
        assert math.isfinite(record.distance)
        assert math.isfinite(record.similarity)
        assert abs(record.similarity - (1 - record.distance)) < 1e-9
    assert collection.count() == count, "Retrieval must not modify the collection"
    print(f"Real Chroma retrieval: PASS ({count} stored, {len(records)} returned; count unchanged)")
    print("Retrieved sources:")
    for record in records:
        print(f"- {record.source}, page {record.page}, {record.chunk_id}, distance={record.distance:.4f}")
    return len(records)


if __name__ == "__main__":
    isolated = run_isolated_tests()
    print(f"Isolated retriever tests: PASS ({isolated})")
    real_results = run_real_chroma_test()
    print(f"Retriever tests completed: {isolated + real_results} checks/results passed")