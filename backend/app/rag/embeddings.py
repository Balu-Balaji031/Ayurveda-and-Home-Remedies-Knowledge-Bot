"""Memory-conscious ONNX embeddings using the existing MiniLM model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
from langchain_core.documents import Document

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMBEDDING_DIMENSIONS = 384
MAX_SEQUENCE_LENGTH = 256
DEFAULT_BATCH_SIZE = 32
ONNX_MODEL_FILE = "onnx/model_quint8_avx2.onnx"
TOKENIZER_FILE = "tokenizer.json"


@dataclass(frozen=True)
class EmbeddedChunk:
    """An embedding kept together with its original text and metadata."""

    text: str
    metadata: dict[str, object]
    vector: list[float]


@lru_cache(maxsize=2)
def _load_runtime(model_name: str) -> tuple[Any, Any, int]:
    """Load a pinned quantized model and its tokenizer once per process."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    revision = EMBEDDING_MODEL_REVISION if model_name == EMBEDDING_MODEL_NAME else None
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=ONNX_MODEL_FILE,
        revision=revision,
    )
    tokenizer_path = hf_hub_download(
        repo_id=model_name,
        filename=TOKENIZER_FILE,
        revision=revision,
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        raise RuntimeError("The embedding tokenizer has no [PAD] token.")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    required_inputs = {"input_ids", "attention_mask", "token_type_ids"}
    actual_inputs = {item.name for item in session.get_inputs()}
    if not required_inputs.issubset(actual_inputs):
        raise RuntimeError("The ONNX embedding model has an unsupported input signature.")
    return tokenizer, session, pad_token_id


def _embed_batch(texts: Sequence[str], *, model_name: str) -> list[list[float]]:
    tokenizer, session, pad_token_id = _load_runtime(model_name)
    encodings = tokenizer.encode_batch(list(texts))
    max_length = max(len(encoding.ids) for encoding in encodings)

    input_ids = np.full((len(encodings), max_length), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((len(encodings), max_length), dtype=np.int64)
    token_type_ids = np.zeros((len(encodings), max_length), dtype=np.int64)
    for row, encoding in enumerate(encodings):
        length = len(encoding.ids)
        input_ids[row, :length] = encoding.ids
        attention_mask[row, :length] = encoding.attention_mask
        token_type_ids[row, :length] = encoding.type_ids

    output_name = session.get_outputs()[0].name
    token_embeddings = session.run(
        [output_name],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]
    mask = attention_mask.astype(np.float32)[..., None]
    pooled = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.clip(norms, 1e-12, None)
    if normalized.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise RuntimeError("The ONNX embedding model returned an unexpected vector shape.")
    if not np.isfinite(normalized).all():
        raise RuntimeError("The ONNX embedding model returned non-finite values.")
    return normalized.astype(np.float32).tolist()


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed text in bounded batches using the existing 384-dimensional model."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty or whitespace-only values")
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        vectors.extend(_embed_batch(texts[start : start + batch_size], model_name=model_name))
    return vectors


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
