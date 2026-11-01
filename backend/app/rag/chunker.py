"""Split extracted PDF pages into bounded, page-aware text chunks."""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_pages(pages: list[Document]) -> list[Document]:
	"""Split pages independently so each chunk retains one precise source page."""
	splitter = RecursiveCharacterTextSplitter(
		chunk_size=CHUNK_SIZE,
		chunk_overlap=CHUNK_OVERLAP,
		add_start_index=True,
	)
	chunks: list[Document] = []

	for page_document in pages:
		source = str(page_document.metadata["source"])
		page = int(page_document.metadata["page"])
		page_chunks = splitter.split_documents([page_document])

		for chunk_index, chunk in enumerate(page_chunks, start=1):
			chunk.page_content = chunk.page_content.strip()
			chunk.metadata.update(
				{
					"source": source,
					"page": page,
					"chunk_id": f"{re.sub(r'[^A-Za-z0-9_-]+', '_', source)}-p{page:04d}-c{chunk_index:04d}",
				}
			)
			if chunk.page_content:
				chunks.append(chunk)

	return chunks