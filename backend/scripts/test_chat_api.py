"""Offline API tests for POST /api/chat; answer generation is always mocked."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient

from app.main import app
from app.rag.answer_generator import AnswerGenerationError, INSUFFICIENT_CONTEXT_ANSWER
from app.rag.gemini_client import (
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiConnectionError,
    GeminiError,
    GeminiModelRequestError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiServerError,
    GeminiTimeoutError,
)
from app.rag.provider_manager import ProviderFallbackError
from app.rag.remedy_workflow import FALLBACK_LABEL


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    @patch("app.api.chat.answer_question")
    def test_success_returns_answer_and_sources(self, answer_question_mock) -> None:
        answer_question_mock.return_value = {
            "answer": "A context-based summary [S1].",
            "sources": [{"source": "guide.pdf", "page": 7, "chunk_id": "guide-p0007-c0001", "snippet": "Source text."}],
            "retrieved_chunk_count": 1,
        }
        response = self.client.post("/api/chat", json={"question": "What does the guide say?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "A context-based summary [S1].",
            "sources": [{"source": "guide.pdf", "page": 7, "chunk_id": "guide-p0007-c0001", "snippet": "Source text."}],
            "error": None,
        })
        answer_question_mock.assert_called_once_with("What does the guide say?")

    def test_empty_and_whitespace_questions_are_rejected(self) -> None:
        for question in ("", "   ", "\n\t"):
            with self.subTest(question=repr(question)):
                response = self.client.post("/api/chat", json={"question": question})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(set(response.json()), {"answer", "sources", "error"})
                self.assertIsNone(response.json()["answer"])
                self.assertEqual(response.json()["sources"], [])
                self.assertTrue(response.json()["error"])

    def test_overlong_question_is_rejected_with_consistent_envelope(self) -> None:
        response = self.client.post("/api/chat", json={"question": "x" * 2001})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(set(response.json()), {"answer", "sources", "error"})
        self.assertIsNone(response.json()["answer"])
        self.assertEqual(response.json()["sources"], [])

    def test_new_workflow_validation_uses_a_general_request_message(self) -> None:
        response = self.client.post("/api/recommendations", json={})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "Request contains invalid or missing fields.")

    @patch("app.api.chat.answer_question")
    def test_insufficient_context_is_a_successful_safe_answer(self, answer_question_mock) -> None:
        answer_question_mock.return_value = {"answer": INSUFFICIENT_CONTEXT_ANSWER, "sources": [], "retrieved_chunk_count": 0}
        response = self.client.post("/api/chat", json={"question": "What does the corpus say?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"answer": INSUFFICIENT_CONTEXT_ANSWER, "sources": [], "error": None})

    @patch("app.api.chat.recommend_remedies")
    def test_recommendations_return_structured_remedies_and_provenance(self, recommend_mock) -> None:
        recommend_mock.return_value = {
            "problem": "cough",
            "mode": "pdf",
            "label": None,
            "remedies": [{
                "id": "pdf:guide-p0001-c0001:1",
                "kind": "pdf",
                "name": "Warm drink",
                "description": "A warm drink may be soothing.",
                "ingredients": None,
                "preparation": None,
                "sources": [{"source": "guide.pdf", "page": 1, "chunk_id": "guide-p0001-c0001", "snippet": "A warm drink may be soothing."}],
            }],
            "sources": [{"source": "guide.pdf", "page": 1, "chunk_id": "guide-p0001-c0001", "snippet": "A warm drink may be soothing."}],
        }
        response = self.client.post("/api/recommendations", json={"problem": "cough"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "pdf")
        self.assertEqual(response.json()["remedies"][0]["sources"][0]["page"], 1)
        self.assertIsNone(response.json()["error"])
        recommend_mock.assert_called_once_with("cough")

    @patch("app.api.chat.recommend_remedies")
    def test_fallback_label_is_returned_without_pdf_sources(self, recommend_mock) -> None:
        recommend_mock.return_value = {
            "problem": "minor discomfort",
            "mode": "ai_generated",
            "label": FALLBACK_LABEL,
            "remedies": [],
            "sources": [],
        }
        response = self.client.post("/api/recommendations", json={"problem": "minor discomfort"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["label"], FALLBACK_LABEL)
        self.assertEqual(response.json()["sources"], [])

    @patch("app.api.chat.explain_remedy")
    def test_remedy_explanation_accepts_selection_and_returns_sources(self, explain_mock) -> None:
        explain_mock.return_value = {
            "answer": "Information is limited.",
            "sources": [{"source": "guide.pdf", "page": 2, "chunk_id": "guide-p0002-c0001", "snippet": "Source passage."}],
        }
        response = self.client.post("/api/remedies/explain", json={
            "problem": "cough",
            "remedy": {
                "id": "pdf:guide-p0002-c0001:1",
                "kind": "pdf",
                "name": "Warm drink",
                "description": "A supportive option.",
                "sources": [{"source": "guide.pdf", "page": 2, "chunk_id": "guide-p0002-c0001", "snippet": "Source passage."}],
            },
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Information is limited.")
        self.assertEqual(response.json()["sources"][0]["source"], "guide.pdf")
        explain_mock.assert_called_once()

    @patch("app.api.chat.answer_follow_up")
    def test_follow_up_passes_selected_remedy_and_history(self, follow_up_mock) -> None:
        follow_up_mock.return_value = {"answer": "The source does not specify.", "sources": []}
        remedy = {
            "id": "ai:1",
            "kind": "ai_generated",
            "name": "Warm drink",
            "description": "Supportive comfort measure.",
            "sources": [],
        }
        response = self.client.post("/api/chat/follow-up", json={
            "problem": "cough",
            "remedy": remedy,
            "question": "Can children use it?",
            "history": [{"question": "What is it?", "answer": "A warm drink."}],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The source does not specify.")
        self.assertEqual(follow_up_mock.call_args.args[0], "cough")
        self.assertEqual(follow_up_mock.call_args.args[2], "Can children use it?")
        self.assertEqual(follow_up_mock.call_args.args[3][0]["question"], "What is it?")

    def test_retrieval_failure_maps_to_service_unavailable(self) -> None:
        response = self._post_error(AnswerGenerationError("Knowledge base unavailable.", category="retrieval"))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "Knowledge base unavailable.")

    def test_gemini_error_categories_map_to_appropriate_http_status(self) -> None:
        errors = [
            (GeminiConfigurationError(("GEMINI_API_KEY",)), 503),
            (GeminiAuthenticationError(403), 502),
            (GeminiModelRequestError(404), 502),
            (GeminiQuotaError(429), 429),
            (GeminiServerError(503), 502),
            (GeminiTimeoutError(504), 504),
            (GeminiConnectionError(), 503),
            (GeminiResponseError(), 502),
            (ProviderFallbackError(), 503),
            (GeminiError(), 500),
        ]
        for error, expected_status in errors:
            with self.subTest(category=error.category, status=error.status_code):
                response = self._post_error(AnswerGenerationError(
                    error.safe_message,
                    category=error.category,
                    status_code=error.status_code,
                ))
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(response.json()["error"], error.safe_message)
                self.assertEqual(set(response.json()), {"answer", "sources", "error"})

    def test_unexpected_error_does_not_expose_internal_text(self) -> None:
        secret_marker = "INTERNAL_PATH_OR_CREDENTIAL_MARKER"
        response = self._post_error(RuntimeError(secret_marker))
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(secret_marker, response.text)
        self.assertEqual(response.json(), {
            "answer": None,
            "sources": [],
            "error": "AyurBot could not process the question. Please try again.",
        })

    @patch("app.api.chat.answer_question")
    def _post_error(self, error: Exception, answer_question_mock=None):
        answer_question_mock.side_effect = error
        return self.client.post("/api/chat", json={"question": "A valid test question"})


if __name__ == "__main__":
    unittest.main(verbosity=2)