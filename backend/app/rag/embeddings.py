"""Reusable local sentence-transformers embeddings for RAG documents and queries."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class EmbeddedChunk:
	"""An embedding kept together with its original text and metadata."""

	text: str
	metadata: dict[str, object]
	vector: list[float]


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
	"""Load each configured model once per process."""
	return SentenceTransformer(model_name, device="cpu")


def embed_texts(
	texts: Sequence[str],
	*,
	model_name: str = EMBEDDING_MODEL_NAME,
	batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
	"""Embed non-empty text in batches, using the same function for queries."""
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")
	if any(not text.strip() for text in texts):
		raise ValueError("texts must not contain empty or whitespace-only values")
	if not texts:
		return []

	vectors = _load_model(model_name).encode(
		list(texts),
		batch_size=batch_size,
		convert_to_numpy=True,
		normalize_embeddings=True,
		show_progress_bar=False,
	)
	return np.asarray(vectors, dtype=np.float32).tolist()


def embed_documents(
	chunks: Sequence[Document],
	*,
	model_name: str = EMBEDDING_MODEL_NAME,
	batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
	"""Embed chunks while preserving their exact text and metadata association."""
	non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
	vectors = embed_texts(
		[chunk.page_content for chunk in non_empty_chunks],
		model_name=model_name,
		batch_size=batch_size,
	)

	if len(vectors) != len(non_empty_chunks):
		raise RuntimeError("The embedding model returned a different number of vectors than inputs")

	return [
		EmbeddedChunk(
			text=chunk.page_content,
			metadata=dict(chunk.metadata),
			vector=vector,
		)
		for chunk, vector in zip(non_empty_chunks, vectors, strict=True)
	]