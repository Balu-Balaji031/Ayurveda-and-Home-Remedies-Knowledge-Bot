"""Extract searchable text from PDFs one page at a time."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_pages(pdf_path: Path) -> list[Document]:
	"""Return non-empty PDF pages as Documents with filename and 1-based page metadata.

	Opening errors are raised so callers can report a failed PDF and continue with
	other files. Page-level extraction errors are logged and skipped.
	"""
	documents: list[Document] = []

	with pymupdf.open(pdf_path) as pdf:
		if pdf.needs_pass:
			raise ValueError("PDF is password-protected and cannot be read without a password")

		for page_index, page in enumerate(pdf, start=1):
			try:
				text = page.get_text("text").strip()
			except Exception as error:
				logger.warning("Could not extract page %s from %s: %s", page_index, pdf_path.name, error)
				continue

			if text:
				documents.append(
					Document(
						page_content=text,
						metadata={"source": pdf_path.name, "page": page_index},
					)
				)

	return documents