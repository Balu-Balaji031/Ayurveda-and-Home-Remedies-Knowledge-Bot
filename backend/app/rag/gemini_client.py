"""Secure, reusable Google Gemini text-generation client."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env", override=False)

DEFAULT_TIMEOUT_MS = 60_000
DEFAULT_TEMPERATURE = 0.2


class GeminiError(RuntimeError):
    """Base class for sanitized Gemini integration errors."""

    category = "unexpected"
    safe_message = "Gemini request failed unexpectedly. Check backend configuration and connectivity."

    def __init__(self, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(self.safe_message)


class GeminiConfigurationError(GeminiError):
    """Required local Gemini configuration is missing or invalid."""

    category = "configuration"
    safe_message = "Gemini is not configured. Set GEMINI_API_KEY and GEMINI_MODEL in backend/.env."

    def __init__(self, missing_variables: tuple[str, ...] = ()) -> None:
        allowed_variables = {"GEMINI_API_KEY", "GEMINI_MODEL"}
        self.missing_variables = tuple(name for name in missing_variables if name in allowed_variables)
        if self.missing_variables:
            self.safe_message = "Gemini is not configured. Set " + " and ".join(self.missing_variables) + " in backend/.env."
        super().__init__()


class GeminiAuthenticationError(GeminiError):
    """Gemini rejected the configured key or project access."""

    category = "authentication"
    safe_message = "Gemini authentication or access was rejected. Check the key and model permissions."


class GeminiModelRequestError(GeminiError):
    """Gemini rejected the requested model or request parameters."""

    category = "model_or_request"
    safe_message = "Gemini rejected the model or request configuration. Check model access and request settings."


class GeminiQuotaError(GeminiError):
    """Gemini quota or rate limits were reached."""

    category = "quota_or_rate_limit"
    safe_message = "Gemini quota or rate limit reached. Try again later or check API quota."


class GeminiServerError(GeminiError):
    """Gemini returned an HTTP 5xx server error."""

    category = "server"
    safe_message = "Gemini returned a server error. Please try again later."


class GeminiTimeoutError(GeminiError):
    """The Gemini request timed out."""

    category = "timeout"
    safe_message = "Gemini did not respond in time. No remedy was generated; please try again."


class GeminiConnectionError(GeminiError):
    """A connection or transport failure prevented reaching Gemini."""

    category = "connection"
    safe_message = "Could not connect to Gemini. Check network connectivity and try again."


class GeminiResponseError(GeminiError):
    """Gemini returned no usable text response."""

    category = "empty_response"
    safe_message = "Gemini returned an empty response. Please try again."


def get_gemini_configuration() -> tuple[str | None, str | None]:
    """Return API key and model from environment without exposing the key."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
    model_name = os.getenv("GEMINI_MODEL", "").strip() or None
    return api_key, model_name


def _require_configuration() -> tuple[str, str]:
    api_key, model_name = get_gemini_configuration()
    missing = [name for name, value in (("GEMINI_API_KEY", api_key), ("GEMINI_MODEL", model_name)) if not value]
    if missing:
        raise GeminiConfigurationError(tuple(missing))
    return api_key, model_name


@lru_cache(maxsize=1)
def _get_client(api_key: str) -> genai.Client:
    """Create and reuse one SDK client for the configured key."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=DEFAULT_TIMEOUT_MS),
    )


def _map_api_error(error: genai_errors.APIError) -> GeminiError:
    status_code = error.code if isinstance(error.code, int) else None
    if status_code == 401:
        return GeminiAuthenticationError(status_code)
    if status_code == 403:
        return GeminiAuthenticationError(status_code)
    if status_code in {400, 404}:
        return GeminiModelRequestError(status_code)
    if status_code == 429:
        return GeminiQuotaError(status_code)
    if status_code in {408, 504}:
        return GeminiTimeoutError(status_code)
    if status_code is not None and status_code >= 500:
        return GeminiServerError(status_code)
    return GeminiModelRequestError(status_code)


def generate_response(
    prompt: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_json_schema: dict[str, Any] | None = None,
    thinking_level: types.ThinkingLevel | None = None,
) -> str:
    """Send one text prompt to Gemini, without retrieval or document context."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")

    api_key, model_name = _require_configuration()
    try:
        response = _get_client(api_key).models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type=response_mime_type,
                response_json_schema=response_json_schema,
                thinking_config=(
                    types.ThinkingConfig(thinking_level=thinking_level)
                    if thinking_level is not None
                    else None
                ),
            ),
        )
    except genai_errors.APIError as error:
        raise _map_api_error(error) from None
    except (httpx.TimeoutException, TimeoutError):
        raise GeminiTimeoutError() from None
    except (httpx.NetworkError, OSError):
        raise GeminiConnectionError() from None
    except Exception:
        raise GeminiError() from None

    generated_text = response.text
    if not isinstance(generated_text, str) or not generated_text.strip():
        raise GeminiResponseError()
    return generated_text.strip()