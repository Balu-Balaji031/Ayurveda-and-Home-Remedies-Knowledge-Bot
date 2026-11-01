"""Persistent ChromaDB storage for AyurBot knowledge-base chunks."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from .embeddings import EMBEDDING_DIMENSIONS, EmbeddedChunk, embed_documents

BACKEND_DIR = Path(__file__).resolve().parents[2]
CHROMA_PERSIST_DIRECTORY = Path(
    os.environ.get(
        "AYURBOT_CHROMA_PATH",
        str(BACKEND_DIR / "data" / "chroma_db"),
    )
).expanduser()
COLLECTION_NAME = "ayurbot_knowledge_base"
MINIMUM_FREE_SPACE_BYTES = 256 * 1024 * 1024
DEFAULT_WRITE_BATCH_SIZE = 64


@dataclass(frozen=True)
class IngestionResult:
    received: int
    embeddings_generated: int
    inserted: int
    updated: int
    unchanged: int
    final_count: int
    persist_directory: str


def _ensure_writable_directory(directory: Path) -> Path:
    directory = directory.resolve()
    parent = directory.parent
    if not parent.exists():
        raise FileNotFoundError(f"ChromaDB parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"ChromaDB parent path is not a directory: {parent}")

    space_path = directory if directory.exists() else parent
    free_space = shutil.disk_usage(space_path).free
    if free_space < MINIMUM_FREE_SPACE_BYTES:
        raise OSError(
            f"Insufficient disk space for ChromaDB at {directory}: "
            f"{free_space} bytes available; at least {MINIMUM_FREE_SPACE_BYTES} required"
        )

    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"ChromaDB path exists but is not a directory: {directory}")

    if directory.exists():
        entries = list(directory.iterdir())
        if entries and not (directory / "chroma.sqlite3").is_file():
            raise RuntimeError(
                f"ChromaDB path contains files but no chroma.sqlite3; refusing to modify existing data: {directory}"
            )
    else:
        directory.mkdir(parents=False, exist_ok=False)

    try:
        with tempfile.NamedTemporaryFile(prefix=".ayurbot_write_check_", dir=directory, delete=True):
            pass
    except OSError as error:
        raise PermissionError(f"ChromaDB directory is not writable: {directory}: {error}") from error

    return directory


def initialize_vector_store(
    persist_directory: Path | str | None = None,
) -> tuple[chromadb.PersistentClient, Collection]:
    """Open or create the persistent client and knowledge-base collection."""
    requested_path = Path(persist_directory) if persist_directory is not None else CHROMA_PERSIST_DIRECTORY
    directory = _ensure_writable_directory(requested_path)
    try:
        client = chromadb.PersistentClient(path=str(directory))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    except Exception as error:
        raise RuntimeError(f"Could not initialize ChromaDB at {directory}: {error}") from error
    return client, collection


def validate_embedded_chunks(records: Sequence[EmbeddedChunk]) -> None:
    record_ids: set[str] = set()
    allowed_metadata_types = (str, int, float, bool)

    for record in records:
        if not isinstance(record.text, str) or not record.text.strip():
            raise ValueError("Document text must be a non-empty string")
        metadata = record.metadata
        required = {"source", "page", "chunk_id"}
        if not required.issubset(metadata):
            raise ValueError(f"Document metadata is missing required keys: {sorted(required - metadata.keys())}")
        if not isinstance(metadata["source"], str) or not metadata["source"].strip():
            raise ValueError("Metadata source must be a non-empty filename")
        if isinstance(metadata["page"], bool) or not isinstance(metadata["page"], int) or metadata["page"] < 1:
            raise ValueError("Metadata page must be a positive 1-based integer")
        chunk_id = metadata["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Metadata chunk_id must be a non-empty string")
        if chunk_id in record_ids:
            raise ValueError(f"Duplicate chunk_id in ingestion input: {chunk_id}")
        record_ids.add(chunk_id)
        if len(record.vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding for {chunk_id} has {len(record.vector)} dimensions; expected {EMBEDDING_DIMENSIONS}"
            )
        for value in record.vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Embedding for {chunk_id} contains a non-finite or non-numeric value")
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, allowed_metadata_types):
                raise ValueError(f"Metadata field {key!r} has a type unsupported by ChromaDB")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Metadata field {key!r} must be finite")


def upsert_embedded_chunks(
    collection: Collection,
    records: Sequence[EmbeddedChunk],
    *,
    batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
) -> int:
    """Validate and upsert embedded chunks in bounded batches; return count written."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    validate_embedded_chunks(records)

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            collection.upsert(
                ids=[str(record.metadata["chunk_id"]) for record in batch],
                documents=[record.text for record in batch],
                embeddings=[record.vector for record in batch],
                metadatas=[record.metadata for record in batch],
            )
        except Exception as error:
            raise RuntimeError(f"ChromaDB write failed for batch starting at record {start}: {error}") from error
    return len(records)


def _existing_records(collection: Collection, ids: Sequence[str]) -> dict[str, tuple[str | None, dict | None]]:
    existing: dict[str, tuple[str | None, dict | None]] = {}
    for start in range(0, len(ids), DEFAULT_WRITE_BATCH_SIZE):
        result = collection.get(
            ids=list(ids[start : start + DEFAULT_WRITE_BATCH_SIZE]),
            include=["documents", "metadatas"],
        )
        for record_id, text, metadata in zip(
            result["ids"], result["documents"] or [], result["metadatas"] or [], strict=True
        ):
            existing[record_id] = (text, metadata)
    return existing


def ingest_chunks(
    chunks: Sequence[Document],
    *,
    collection: Collection,
    persist_directory: Path | str = CHROMA_PERSIST_DIRECTORY,
    embedding_batch_size: int = 16,
    write_batch_size: int = DEFAULT_WRITE_BATCH_SIZE,
) -> IngestionResult:
    """Embed only new/changed chunks and idempotently upsert them by chunk_id."""
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be at least 1")
    if write_batch_size < 1:
        raise ValueError("write_batch_size must be at least 1")

    chunk_by_id: dict[str, Document] = {}
    for chunk in chunks:
        if not chunk.page_content.strip():
            raise ValueError("Chunk text must not be empty")
        metadata = chunk.metadata
        validate_embedded_chunks(
            [EmbeddedChunk(text=chunk.page_content, metadata=dict(metadata), vector=[0.0] * EMBEDDING_DIMENSIONS)]
        )
        chunk_id = str(metadata["chunk_id"])
        if chunk_id in chunk_by_id:
            raise ValueError(f"Duplicate chunk_id in ingestion input: {chunk_id}")
        chunk_by_id[chunk_id] = chunk

    ids = list(chunk_by_id)
    existing = _existing_records(collection, ids) if ids else {}
    changed_chunks = [
        chunk
        for chunk_id, chunk in chunk_by_id.items()
        if chunk_id not in existing
        or existing[chunk_id][0] != chunk.page_content
        or existing[chunk_id][1] != chunk.metadata
    ]

    try:
        embedded = embed_documents(changed_chunks, batch_size=embedding_batch_size) if changed_chunks else []
    except Exception as error:
        raise RuntimeError(f"Embedding generation failed: {error}") from error

    written = upsert_embedded_chunks(collection, embedded, batch_size=write_batch_size) if embedded else 0
    inserted = sum(1 for item in embedded if str(item.metadata["chunk_id"]) not in existing)
    updated = written - inserted
    unchanged = len(chunk_by_id) - len(changed_chunks)
    return IngestionResult(
        received=len(chunks),
        embeddings_generated=len(embedded),
        inserted=inserted,
        updated=updated,
        unchanged=unchanged,
        final_count=collection.count(),
        persist_directory=str(Path(persist_directory).expanduser().resolve()),
    )


def get_collection_stats(collection: Collection, persist_directory: Path | str | None = None) -> dict[str, object]:
    """Return the current knowledge collection name, count, and storage path."""
    path = Path(persist_directory) if persist_directory is not None else CHROMA_PERSIST_DIRECTORY
    return {
        "collection_name": collection.name,
        "count": collection.count(),
        "persist_directory": str(path.expanduser().resolve()),
    }