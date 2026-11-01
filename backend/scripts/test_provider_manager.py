"""Mocked tests for the Gemini-primary, Groq-fallback provider manager."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import httpx

from app.rag import answer_generator, provider_manager, remedy_workflow
from app.rag.gemini_client import (
    GeminiAuthenticationError,
    GeminiConnectionError,
    GeminiModelRequestError,
    GeminiQuotaError,
    GeminiServerError,
    GeminiTimeoutError,
)
from app.rag.provider_manager import ProviderFallbackError
from app.rag.retriever import RetrievedPassage, RetrievalResult


class ProviderManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test-key-not-real", "GROQ_MODEL": "openai/gpt-oss-20b"},
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_gemini_success_does_not_call_groq(self) -> None:
        with patch.object(provider_manager, "generate_gemini_response", return_value="Gemini answer"):
            with patch.object(provider_manager, "_call_groq") as fallback:
                self.assertEqual(provider_manager.generate_response("same prompt"), "Gemini answer")
            fallback.assert_not_called()

    def test_quota_server_and_timeout_errors_trigger_one_groq_fallback(self) -> None:
        failures = (
            GeminiQuotaError(429),
            GeminiServerError(503),
            GeminiTimeoutError(),
            GeminiConnectionError(),
            GeminiServerError(500),
            GeminiServerError(502),
            GeminiServerError(504),
        )
        for failure in failures:
            with self.subTest(category=failure.category, status=failure.status_code):
                with patch.object(provider_manager, "generate_gemini_response", side_effect=failure):
                    with patch.object(provider_manager, "_call_groq", return_value="Groq answer") as fallback:
                        self.assertEqual(
                            provider_manager.generate_response("same prompt", max_output_tokens=400),
                            "Groq answer",
                        )
                self.assertEqual(fallback.call_count, 1)
                self.assertEqual(fallback.call_args.args[0], "same prompt")
                self.assertEqual(fallback.call_args.kwargs["max_output_tokens"], 400)

    def test_gemini_authentication_error_does_not_call_groq(self) -> None:
        with patch.object(
            provider_manager,
            "generate_gemini_response",
            side_effect=GeminiAuthenticationError(401),
        ):
            with patch.object(provider_manager, "_call_groq") as fallback:
                with self.assertRaises(GeminiAuthenticationError):
                    provider_manager.generate_response("same prompt")
        fallback.assert_not_called()

    def test_permanent_request_errors_do_not_call_groq(self) -> None:
        for failure in (GeminiModelRequestError(400), GeminiServerError(501)):
            with self.subTest(category=failure.category, status=failure.status_code):
                with patch.object(provider_manager, "generate_gemini_response", side_effect=failure):
                    with patch.object(provider_manager, "_call_groq") as fallback:
                        with self.assertRaises(type(failure)):
                            provider_manager.generate_response("same prompt")
                fallback.assert_not_called()

    def test_groq_chat_completion_uses_configured_model_and_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"answer":"safe"}'}}],
        }
        with patch.object(provider_manager.httpx, "post", return_value=response) as post:
            answer = provider_manager._call_groq(
                "verified PDF context",
                temperature=0.2,
                max_output_tokens=400,
                response_mime_type="application/json",
                response_json_schema=schema,
            )
        self.assertEqual(answer, '{"answer":"safe"}')
        self.assertEqual(post.call_args.args[0], "https://api.groq.com/openai/v1/chat/completions")
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["model"], "openai/gpt-oss-20b")
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["messages"], [{"role": "user", "content": "verified PDF context"}])
        self.assertEqual(request["max_completion_tokens"], 400)
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIsNotNone(schema)
        self.assertEqual(post.call_args.kwargs["timeout"], provider_manager.GROQ_TIMEOUT_SECONDS)
        self.assertNotIn("test-key-not-real", str(request))
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key-not-real")

    def test_groq_invalid_key_response_is_classified_as_authentication(self) -> None:
        response = Mock(status_code=400)
        response.json.return_value = {"error": {"message": "Incorrect API key provided."}}
        with patch.object(provider_manager.httpx, "post", return_value=response):
            with self.assertRaises(GeminiAuthenticationError) as raised:
                provider_manager._call_groq(
                    "connectivity check",
                    temperature=0,
                    max_output_tokens=40,
                    response_mime_type=None,
                    response_json_schema=None,
                )
        self.assertEqual(raised.exception.category, "authentication")
        self.assertEqual(raised.exception.status_code, 400)
        self.assertNotIn("Incorrect API key", str(raised.exception))

    def test_both_providers_failing_returns_only_friendly_error(self) -> None:
        secret = "sensitive response text and key"
        with patch.object(provider_manager, "generate_gemini_response", side_effect=GeminiQuotaError(429)):
            with patch.object(provider_manager, "_call_groq", side_effect=httpx.ReadTimeout(secret)):
                with self.assertRaises(ProviderFallbackError) as raised:
                    provider_manager.generate_response("private user prompt")
        self.assertEqual(
            str(raised.exception),
            "AyurBot's AI services are temporarily unavailable. Please try again later.",
        )
        self.assertNotIn(secret, str(raised.exception))

    def test_groq_not_configured_returns_friendly_error_without_request(self) -> None:
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "GROQ_MODEL": ""}):
            with patch.object(provider_manager, "generate_gemini_response", side_effect=GeminiQuotaError(429)):
                with patch.object(provider_manager.httpx, "post") as post:
                    with self.assertRaises(ProviderFallbackError):
                        provider_manager.generate_response("prompt")
        post.assert_not_called()

    def test_gemini_to_groq_fallback_preserves_pdf_sources(self) -> None:
        result = RetrievalResult(
            text="A verified cough passage.",
            source="cough-guide.pdf",
            page=8,
            chunk_id="cough-p0008-c0001",
            distance=0.1,
            similarity=0.9,
        )
        with patch.object(answer_generator, "retrieve_documents", return_value=[result]):
            with patch.object(
                provider_manager,
                "generate_gemini_response",
                side_effect=GeminiQuotaError(429),
            ):
                with patch.object(provider_manager, "_call_groq", return_value="Grounded Groq answer [S1]."):
                    response = answer_generator.answer_question("How may the PDF help?")
        self.assertEqual(response["answer"], "Grounded Groq answer [S1].")
        self.assertEqual(response["sources"][0]["source"], "cough-guide.pdf")
        self.assertEqual(response["sources"][0]["page"], 8)
        self.assertIn("A verified cough passage.", response["sources"][0]["snippet"])

    def test_follow_up_fallback_preserves_selected_remedy_history_and_sources(self) -> None:
        passage = RetrievedPassage(
            text="Warm water sips may soothe a dry cough.",
            source="cough-guide.pdf",
            page=8,
            chunk_id="cough-p0008-c0001",
        )
        remedy = {
            "id": "pdf:warm-water",
            "kind": "pdf",
            "name": "Warm water sips",
            "description": "Warm water sips may soothe a dry cough.",
            "sources": [{
                "source": passage.source,
                "page": passage.page,
                "chunk_id": passage.chunk_id,
                "snippet": passage.text,
            }],
        }
        additional = SimpleNamespace(
            text=passage.text,
            source=passage.source,
            page=passage.page,
            chunk_id=passage.chunk_id,
        )
        history = [{"question": "How do I prepare it?", "answer": "Sip warm water."}]
        with patch.object(remedy_workflow, "retrieve_documents_by_ids", return_value=[passage]):
            with patch.object(remedy_workflow, "retrieve_documents", return_value=[additional]):
                with patch.object(
                    provider_manager,
                    "generate_gemini_response",
                    side_effect=GeminiTimeoutError(),
                ):
                    with patch.object(provider_manager, "_call_groq", return_value="Use comfortable warmth.") as fallback:
                        result = remedy_workflow.answer_follow_up(
                            "dry cough",
                            remedy,
                            "Can children use this?",
                            history,
                        )
        prompt = fallback.call_args.args[0]
        self.assertIn("Warm water sips", prompt)
        self.assertIn("Can children use this?", prompt)
        self.assertIn("How do I prepare it?", prompt)
        self.assertIn(passage.text, prompt)
        self.assertEqual(result["answer"], "Use comfortable warmth.")
        self.assertEqual(result["sources"][0]["source"], passage.source)
        self.assertEqual(result["sources"][0]["page"], passage.page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
