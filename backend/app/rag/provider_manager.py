"""Primary Gemini generation with a bounded Groq fallback."""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import httpx
from google.genai import types

from .gemini_client import (
    DEFAULT_TEMPERATURE,
    GeminiAuthenticationError,
    GeminiConfigurationError,
    GeminiConnectionError,
    GeminiError,
    GeminiModelRequestError,
    GeminiQuotaError,
    GeminiResponseError,
    GeminiServerError,
    GeminiTimeoutError,
    generate_response as generate_gemini_response,
)

logger = logging.getLogger(__name__)
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT_SECONDS = 30


class ProviderFallbackError(GeminiError):
    """Safe error raised when neither configured text provider can answer."""

    category = "provider_unavailable"
    safe_message = "AyurBot's AI services are temporarily unavailable. Please try again later."


def _groq_configuration() -> tuple[str, str]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip()
    if not api_key or not model:
        missing = [
            name for name, value in (("GROQ_API_KEY", api_key), ("GROQ_MODEL", model))
            if not value
        ]
        logger.warning(
            "Provider fallback unavailable: provider=groq category=configuration missing=%s",
            ",".join(missing),
        )
        raise ProviderFallbackError()
    return api_key, model


def _retryable_gemini_error(error: GeminiError) -> bool:
    if isinstance(error, GeminiQuotaError):
        return True
    if isinstance(error, GeminiTimeoutError):
        return error.status_code in {None, 408, 504}
    if isinstance(error, GeminiConnectionError):
        return True
    return isinstance(error, GeminiServerError) and error.status_code in {500, 502, 503, 504}


def _is_groq_authentication_error(response: httpx.Response) -> bool:
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message", "") if isinstance(error, dict) else error
    if not isinstance(message, str):
        return False
    normalized = message.casefold()
    return any(
        phrase in normalized
        for phrase in ("incorrect api key", "invalid api key", "api key is invalid")
    )


def _groq_request(
    prompt: str,
    *,
    api_key: str,
    model: str,
    temperature: float,
    max_output_tokens: int | None,
    response_mime_type: str | None,
    response_json_schema: dict[str, Any] | None,
) -> str:
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": min(temperature, 1.0),
        "stream": False,
    }
    # GPT-OSS defaults to medium reasoning, which can consume the whole completion budget.
    if model.casefold().startswith("openai/gpt-oss-"):
        request["reasoning_effort"] = "low"
    if max_output_tokens is not None:
        request["max_completion_tokens"] = max_output_tokens
    if response_mime_type == "application/json":
        # Groq strict schemas require every property; AyurBot's schema has optional fields.
        request["response_format"] = {"type": "json_object"}

    response = httpx.post(
        f"{GROQ_API_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request,
        timeout=GROQ_TIMEOUT_SECONDS,
    )
    if response.status_code == 400 and _is_groq_authentication_error(response):
        raise GeminiAuthenticationError(response.status_code)
    if response.status_code in {401, 403}:
        raise GeminiAuthenticationError(response.status_code)
    if response.status_code == 429:
        raise GeminiQuotaError(response.status_code)
    if response.status_code in {408, 504}:
        raise GeminiTimeoutError(response.status_code)
    if response.status_code in {500, 502, 503}:
        raise GeminiServerError(response.status_code)
    if response.status_code >= 400:
        raise GeminiModelRequestError(response.status_code)

    try:
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise GeminiResponseError() from None
    if not isinstance(content, str) or not content.strip():
        raise GeminiResponseError()
    return content.strip()


def _call_groq(
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int | None,
    response_mime_type: str | None,
    response_json_schema: dict[str, Any] | None,
) -> str:
    api_key, model = _groq_configuration()
    return _groq_request(
        prompt,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_json_schema=response_json_schema,
    )


def _groq_error_category(error: Exception) -> tuple[str, int | None]:
    if isinstance(error, GeminiError):
        return error.category, error.status_code
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "timeout", None
    if isinstance(error, (httpx.NetworkError, OSError)):
        return "connection", None
    return "unexpected", None


def generate_response(
    prompt: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    response_json_schema: dict[str, Any] | None = None,
    thinking_level: types.ThinkingLevel | None = None,
) -> str:
    """Generate with Gemini first, falling back once to Groq on temporary failures."""
    started = perf_counter()
    try:
        answer = generate_gemini_response(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            response_json_schema=response_json_schema,
            thinking_level=thinking_level,
        )
    except GeminiConfigurationError:
        logger.warning(
            "Text generation failed: provider=gemini category=configuration fallback_used=false duration_ms=%d",
            round((perf_counter() - started) * 1000),
        )
        raise
    except GeminiError as error:
        if not _retryable_gemini_error(error):
            logger.warning(
                "Text generation failed: provider=gemini category=%s status_code=%s fallback_used=false duration_ms=%d",
                error.category,
                error.status_code,
                round((perf_counter() - started) * 1000),
            )
            raise

        logger.warning(
            "Text provider fallback triggered: primary=gemini category=%s status_code=%s "
            "fallback=groq primary_duration_ms=%d",
            error.category,
            error.status_code,
            round((perf_counter() - started) * 1000),
        )
    else:
        logger.info(
            "Text generation succeeded: provider=gemini fallback_used=false duration_ms=%d",
            round((perf_counter() - started) * 1000),
        )
        return answer

    groq_started = perf_counter()
    try:
        answer = _call_groq(
            prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            response_json_schema=response_json_schema,
        )
    except Exception as error:
        category, status_code = _groq_error_category(error)
        logger.warning(
            "Text generation failed: provider=groq category=%s status_code=%s fallback_used=true duration_ms=%d",
            category,
            status_code,
            round((perf_counter() - groq_started) * 1000),
        )
        raise ProviderFallbackError() from None

    logger.info(
        "Text generation succeeded: provider=groq fallback_used=true duration_ms=%d",
        round((perf_counter() - groq_started) * 1000),
    )
    return answer
