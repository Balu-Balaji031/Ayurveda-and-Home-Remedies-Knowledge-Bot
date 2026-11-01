"""Test context-grounded answer construction and safe failure behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag import answer_generator
from app.rag.answer_generator import (
    INSUFFICIENT_CONTEXT_ANSWER,
    AnswerGenerationError,
    answer_question,
)
from app.rag.gemini_client import GeminiAuthenticationError, GeminiServerError
from app.rag.retriever import RetrievalError, RetrievalResult


def run_tests() -> int:
    checks = 0
    first = RetrievalResult(
        text="The source notes that preparation methods vary.",
        source="Ayurvedic-Home-Remedies-English.pdf",
        page=12,
        chunk_id="herbs-p0012-c0001",
        distance=0.2,
        similarity=0.8,
    )
    second = RetrievalResult(
        text="Consult an appropriate professional if symptoms worsen.",
        source="Evidence_based_Ayurvedic_Practice.pdf",
        page=24,
        chunk_id="practice-p0024-c0003",
        distance=0.3,
        similarity=0.7,
    )

    gemini = Mock(return_value="A grounded answer [S1].")
    with patch.object(answer_generator, "retrieve_documents", return_value=[first, second]) as retrieve:
        with patch.object(answer_generator, "generate_response", gemini):
            response = answer_question("How do the documents discuss remedies?")
    prompt = gemini.call_args.args[0]
    assert retrieve.call_args.args[0] == "How do the documents discuss remedies?"
    assert "The source notes that preparation methods vary." in prompt
    assert "Consult an appropriate professional if symptoms worsen." in prompt
    assert "untrusted retrieved document passages" in prompt
    assert "How do the documents discuss remedies?" in prompt
    assert response["answer"] == "A grounded answer [S1]."
    assert response["sources"] == [
        {"source": first.source, "page": first.page, "chunk_id": first.chunk_id, "snippet": first.text},
        {"source": second.source, "page": second.page, "chunk_id": second.chunk_id, "snippet": second.text},
    ]
    checks += 1

    with patch.object(answer_generator, "retrieve_documents", return_value=[first, first]):
        with patch.object(answer_generator, "generate_response", return_value="A cited response."):
            deduped = answer_question("Question")
    assert len(deduped["sources"]) == 1
    checks += 1

    with patch.object(answer_generator, "retrieve_documents", return_value=[]):
        with patch.object(answer_generator, "generate_response") as no_context_gemini:
            no_context = answer_question("Question with no source context")
    assert no_context["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert no_context["sources"] == []
    assert not no_context_gemini.called
    checks += 1

    with patch.object(answer_generator, "retrieve_documents", side_effect=RetrievalError("store unavailable")):
        with patch.object(answer_generator, "generate_response") as retrieval_failure_gemini:
            try:
                answer_question("Question")
            except AnswerGenerationError as error:
                assert "could not access its knowledge base" in str(error)
                assert not retrieval_failure_gemini.called
                checks += 1
            else:
                raise AssertionError("Retrieval failure was not raised safely")

    with patch.object(answer_generator, "retrieve_documents", return_value=[first]):
        with patch.object(answer_generator, "generate_response", side_effect=GeminiAuthenticationError(403)):
            try:
                answer_question("Question")
            except AnswerGenerationError as error:
                assert "Gemini authentication or access was rejected" in str(error)
                assert error.category == "authentication"
                assert error.status_code == 403
                checks += 1
            else:
                raise AssertionError("Gemini error was not raised safely")

    with patch.object(answer_generator, "retrieve_documents", return_value=[first]):
        with patch.object(answer_generator, "generate_response", side_effect=GeminiServerError(503)):
            try:
                answer_question("Question")
            except AnswerGenerationError as error:
                assert error.category == "server"
                assert error.status_code == 503
                checks += 1
            else:
                raise AssertionError("Unexpected failure was not wrapped")

    try:
        answer_question(" ")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Empty question was not rejected")

    return checks


if __name__ == "__main__":
    passed = run_tests()
    print(f"Answer-generator tests: PASS ({passed} checks)")