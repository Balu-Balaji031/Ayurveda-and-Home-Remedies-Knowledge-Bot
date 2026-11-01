"""Safely validate Gemini configuration, error handling, and optionally one live call."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import httpx
from google.genai import errors as genai_errors

from app.rag import gemini_client


def run_mocked_checks() -> int:
    checks = 0
    assert gemini_client.DEFAULT_TIMEOUT_MS == 60_000
    checks += 1

    try:
        gemini_client.generate_response("   ")
    except ValueError:
        checks += 1
    else:
        raise AssertionError("Empty prompt was not rejected")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": "test-model"}, clear=False):
        try:
            gemini_client.generate_response("A harmless test prompt")
        except gemini_client.GeminiConfigurationError as error:
            assert "GEMINI_API_KEY" in str(error)
            checks += 1
        else:
            raise AssertionError("Missing API key was not rejected")

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-not-real", "GEMINI_MODEL": ""}, clear=False):
        try:
            gemini_client.generate_response("A harmless test prompt")
        except gemini_client.GeminiConfigurationError as error:
            assert "GEMINI_MODEL" in str(error)
            assert "test-key-not-real" not in str(error)
            checks += 1
        else:
            raise AssertionError("Missing model was not rejected")

    api_failures = [
        (genai_errors.ClientError(401, {"message": "sensitive-test-value"}), gemini_client.GeminiAuthenticationError, "authentication", 401),
        (genai_errors.ClientError(403, {"message": "sensitive-test-value"}), gemini_client.GeminiAuthenticationError, "authentication", 403),
        (genai_errors.ClientError(400, {"message": "sensitive-test-value"}), gemini_client.GeminiModelRequestError, "model_or_request", 400),
        (genai_errors.ClientError(404, {"message": "sensitive-test-value"}), gemini_client.GeminiModelRequestError, "model_or_request", 404),
        (genai_errors.ClientError(429, {"message": "sensitive-test-value"}), gemini_client.GeminiQuotaError, "quota_or_rate_limit", 429),
        (genai_errors.ClientError(504, {"message": "sensitive-test-value"}), gemini_client.GeminiTimeoutError, "timeout", 504),
        (genai_errors.ServerError(500, {"message": "sensitive-test-value"}), gemini_client.GeminiServerError, "server", 500),
        (genai_errors.ServerError(503, {"message": "sensitive-test-value"}), gemini_client.GeminiServerError, "server", 503),
        (httpx.ReadTimeout("sensitive-test-value"), gemini_client.GeminiTimeoutError, "timeout", None),
        (httpx.ConnectError("sensitive-test-value"), gemini_client.GeminiConnectionError, "connection", None),
    ]
    for sdk_error, expected_type, category, status_code in api_failures:
        client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(side_effect=sdk_error)))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-not-real", "GEMINI_MODEL": "test-model"}, clear=False):
            with patch.object(gemini_client, "_get_client", return_value=client):
                try:
                    gemini_client.generate_response("A harmless test prompt")
                except expected_type as error:
                    assert "sensitive-test-value" not in str(error)
                    assert "test-key-not-real" not in str(error)
                    assert error.category == category
                    assert error.status_code == status_code
                    checks += 1
                else:
                    raise AssertionError(f"Expected {expected_type.__name__} for a mocked SDK failure")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text="  test response  "))))
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-not-real", "GEMINI_MODEL": "test-model"}, clear=False):
        with patch.object(gemini_client, "_get_client", return_value=client):
            assert gemini_client.generate_response("A harmless test prompt") == "test response"
            checks += 1

    client = SimpleNamespace(models=SimpleNamespace(generate_content=Mock(return_value=SimpleNamespace(text=" "))))
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-not-real", "GEMINI_MODEL": "test-model"}, clear=False):
        with patch.object(gemini_client, "_get_client", return_value=client):
            try:
                gemini_client.generate_response("A harmless test prompt")
            except gemini_client.GeminiResponseError:
                checks += 1
            else:
                raise AssertionError("Empty Gemini response was not rejected")

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="run local mocked checks only; do not send a request to Gemini",
    )
    args = parser.parse_args()

    print("Gemini SDK import: PASS")
    api_key, model_name = gemini_client.get_gemini_configuration()
    print(f"GEMINI_API_KEY configured: {'yes' if api_key else 'no'}")
    print(f"GEMINI_MODEL configured: {'yes' if model_name else 'no'}")
    if model_name:
        print(f"Selected model: {model_name}")
    else:
        print("Selected model: not selected (set GEMINI_MODEL to a model enabled for this key)")

    mocked_checks = run_mocked_checks()
    print(f"Mocked configuration/error/success checks: PASS ({mocked_checks})")

    if args.no_live:
        print("Live Gemini request: SKIPPED (--no-live specified)")
        return 0

    if not api_key or not model_name:
        print("Live Gemini request: SKIPPED (GEMINI_API_KEY and GEMINI_MODEL must both be configured)")
        return 0

    try:
        response = gemini_client.generate_response(
            "Reply with a short confirmation that this Gemini connectivity test succeeded."
        )
    except gemini_client.GeminiError as error:
        print(f"Live Gemini request: FAILED ({error})")
        return 1

    print("Live Gemini request: PASS")
    print(f"Response preview: {response[:160].replace(chr(10), ' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())