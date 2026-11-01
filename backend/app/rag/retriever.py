"""Semantic retrieval from AyurBot's existing persistent Chroma collection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from .embeddings import EMBEDDING_DIMENSIONS, embed_texts
from .vector_store import CHROMA_PERSIST_DIRECTORY, COLLECTION_NAME

DEFAULT_RETRIEVAL_COUNT = 5


class RetrievalError(RuntimeError):
    """Base class for safe knowledge retrieval failures."""


class KnowledgeBaseUnavailableError(RetrievalError):
    """The persistent database or expected collection is unavailable."""


class QueryEmbeddingError(RetrievalError):
    """The question could not be represented as a valid query vector."""


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    source: str
    page: int
    chunk_id: str
    distance: float
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "distance": self.distance,
            "similarity": self.similarity,
        }


@dataclass(frozen=True)
class RetrievedPassage:
    text: str
    source: str
    page: int
    chunk_id: str


@lru_cache(maxsize=1)
def _open_existing_collection() -> Collection:
    """Open the configured collection without creating or modifying anything."""
    directory = Path(CHROMA_PERSIST_DIRECTORY).expanduser().resolve()
    if not directory.is_dir() or not (directory / "chroma.sqlite3").is_file():
        raise KnowledgeBaseUnavailableError(
            f"The ChromaDB persistent database is missing at {directory}. Run document ingestion first."
        )
    try:
        client = chromadb.PersistentClient(path=str(directory))
        return client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    except Exception as error:
        if isinstance(error, chromadb.errors.NotFoundError):
            raise KnowledgeBaseUnavailableError(
                f"The ChromaDB collection '{COLLECTION_NAME}' was not found. Run document ingestion first."
            ) from None
        raise RetrievalError("Could not open the existing ChromaDB collection.") from None


def retrieve_documents_by_ids(chunk_ids: list[str]) -> list[RetrievedPassage]:
    """Load exact existing passages by ID without changing the collection."""
    if not chunk_ids or any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in chunk_ids):
        raise ValueError("chunk_ids must contain non-empty strings")
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_ids must be unique")

    collection = _open_existing_collection()
    try:
        result = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    except Exception:
        raise RetrievalError("ChromaDB could not retrieve the requested source passages.") from None

    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []
    if len(texts) != len(metadatas) or len(texts) != len(ids):
        raise RetrievalError("ChromaDB returned incomplete source passage records.")

    passages: dict[str, RetrievedPassage] = {}
    for text, metadata, chunk_id in zip(texts, metadatas, ids, strict=True):
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
            raise RetrievalError("ChromaDB returned an invalid source passage.")
        source = metadata.get("source")
        page = metadata.get("page")
        if not isinstance(source, str) or not source.strip():
            raise RetrievalError("A source passage is missing its filename.")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise RetrievalError("A source passage has an invalid page number.")
        if metadata.get("chunk_id") != chunk_id:
            raise RetrievalError("A source passage has inconsistent chunk metadata.")
        passages[chunk_id] = RetrievedPassage(text=text, source=source, page=page, chunk_id=chunk_id)

    if set(passages) != set(chunk_ids):
        raise RetrievalError("One or more requested source passages were not found.")
    return [passages[chunk_id] for chunk_id in chunk_ids]


def _validate_query_vector(vector: Any) -> list[float]:
    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError, OverflowError):
        raise QueryEmbeddingError("The question embedding is invalid.") from None
    if len(values) != EMBEDDING_DIMENSIONS:
        raise QueryEmbeddingError(
            f"The question embedding has {len(values)} dimensions; expected {EMBEDDING_DIMENSIONS}."
        )
    if not all(math.isfinite(value) for value in values):
        raise QueryEmbeddingError("The question embedding contains non-finite values.")
    return values


def retrieve_documents(
    question: str,
    *,
    retrieval_count: int = DEFAULT_RETRIEVAL_COUNT,
    collection: Collection | None = None,
) -> list[RetrievalResult]:
    """Embed a question and retrieve its nearest stored document chunks.

    Chroma uses cosine distance for the existing collection. Similarity is
    explicitly converted as ``1 - distance``; the raw distance is also kept.
    ``collection`` is injectable for isolated tests and does not change normal
    use of the existing configured collection.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if retrieval_count < 1:
        raise ValueError("retrieval_count must be at least 1")

    try:
        vectors = embed_texts([question.strip()])
    except Exception:
        raise QueryEmbeddingError("Could not generate an embedding for the question.") from None
    if len(vectors) != 1:
        raise QueryEmbeddingError("The embedding model did not return exactly one query vector.")
    query_vector = _validate_query_vector(vectors[0])

    target_collection = collection if collection is not None else _open_existing_collection()
    try:
        collection_count = target_collection.count()
        if collection_count == 0:
            return []
        result = target_collection.query(
            query_embeddings=[query_vector],
            n_results=min(retrieval_count, collection_count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        raise RetrievalError("ChromaDB could not retrieve knowledge-base documents.") from None

    texts = (result.get("documents") or [[]])[0] or []
    metadatas = (result.get("metadatas") or [[]])[0] or []
    distances = (result.get("distances") or [[]])[0] or []
    ids = (result.get("ids") or [[]])[0] or []

    if not (len(texts) == len(metadatas) == len(distances) == len(ids)):
        raise RetrievalError("ChromaDB returned incomplete retrieval records.")

    retrieved: list[RetrievalResult] = []
    for text, metadata, raw_distance, record_id in zip(texts, metadatas, distances, ids, strict=True):
        if not isinstance(text, str) or not text.strip():
            raise RetrievalError("ChromaDB returned an empty document chunk.")
        if not isinstance(metadata, dict):
            raise RetrievalError("ChromaDB returned missing document metadata.")
        source = metadata.get("source")
        page = metadata.get("page")
        chunk_id = metadata.get("chunk_id")
        if not isinstance(source, str) or not source.strip():
            raise RetrievalError("Retrieved document is missing its source filename.")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise RetrievalError("Retrieved document has an invalid 1-based page number.")
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id != record_id:
            raise RetrievalError("Retrieved document has missing or inconsistent chunk_id metadata.")
        try:
            distance = float(raw_distance)
        except (TypeError, ValueError, OverflowError):
            raise RetrievalError("ChromaDB returned an invalid cosine distance.") from None
        if not math.isfinite(distance):
            raise RetrievalError("ChromaDB returned a non-finite cosine distance.")
        retrieved.append(
            RetrievalResult(
                text=text,
                source=source,
                page=page,
                chunk_id=chunk_id,
                distance=distance,
                similarity=1.0 - distance,
            )
        )
    return retrieved