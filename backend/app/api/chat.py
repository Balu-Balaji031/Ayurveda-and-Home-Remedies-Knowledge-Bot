"""Chat API bridge to the existing retrieval and answer-generation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from app.rag.answer_generator import AnswerGenerationError, answer_question
from app.rag.gemini_client import GeminiError
from app.rag.remedy_workflow import (
    RemedyWorkflowError,
    answer_follow_up,
    explain_remedy,
    recommend_remedies,
)
from app.rag.retriever import RetrievalError

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_QUESTION_LENGTH = 2_000


class ChatRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_LENGTH)


class RemedySource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=260)
    page: int = Field(ge=1)
    chunk_id: str = Field(min_length=1, max_length=320)
    snippet: str = Field(min_length=1, max_length=2_000)


class RemedyCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=400)
    kind: Literal["pdf", "ai_generated"]
    name: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=3_000)
    ingredients: str | None = Field(default=None, max_length=2_000)
    preparation: str | None = Field(default=None, max_length=2_000)
    traditional_or_supportive_basis: str | None = Field(default=None, max_length=2_000)
    evidence: str | None = Field(default=None, max_length=2_000)
    possible_benefits: str | None = Field(default=None, max_length=2_000)
    side_effects: str | None = Field(default=None, max_length=2_000)
    who_should_avoid: str | None = Field(default=None, max_length=2_000)
    precautions_interactions: str | None = Field(default=None, max_length=2_000)
    when_to_seek_care: str | None = Field(default=None, max_length=2_000)
    sources: list[RemedySource] = Field(default_factory=list, max_length=5)


class RecommendationRequest(BaseModel):
    problem: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class RemedyActionRequest(BaseModel):
    problem: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    remedy: RemedyCard


class FollowUpTurn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    answer: str = Field(min_length=1, max_length=6_000)


class FollowUpRequest(RemedyActionRequest):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    history: list[FollowUpTurn] = Field(default_factory=list, max_length=8)


def _error_status(category: str) -> int:
    return {
        "configuration": 503,
        "retrieval": 503,
        "connection": 503,
        "provider_unavailable": 503,
        "authentication": 502,
        "model_or_request": 502,
        "quota_or_rate_limit": 429,
        "server": 502,
        "timeout": 504,
        "empty_response": 502,
    }.get(category, 500)


def _workflow_error(error: Exception) -> JSONResponse:
    if isinstance(error, GeminiError):
        logger.warning(
            "AI provider remedy workflow failed: category=%s status_code=%s",
            error.category,
            error.status_code,
        )
        return error_response(error.safe_message, _error_status(error.category))
    if isinstance(error, RetrievalError):
        logger.warning("Remedy workflow retrieval failed (%s)", type(error).__name__)
        return error_response("AyurBot could not access its knowledge base. Please try again later.", 503)
    if isinstance(error, RemedyWorkflowError):
        logger.warning("Remedy workflow response could not be validated")
        return error_response(str(error), 502)
    if isinstance(error, ValueError):
        return error_response(str(error), 422)
    logger.error("Unexpected remedy workflow failure (%s)", type(error).__name__)
    return error_response("AyurBot could not process the request. Please try again.", 500)


def error_response(message: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"answer": None, "sources": [], "error": message},
    )


@router.post("/recommendations", response_model=None)
async def recommendations(request: RecommendationRequest) -> dict[str, Any] | JSONResponse:
    try:
        result = recommend_remedies(request.problem)
    except Exception as error:
        return _workflow_error(error)
    return {**result, "error": None}


@router.post("/remedies/explain", response_model=None)
async def remedy_explanation(request: RemedyActionRequest) -> dict[str, Any] | JSONResponse:
    try:
        result = explain_remedy(request.problem, request.remedy.model_dump())
    except Exception as error:
        return _workflow_error(error)
    return {**result, "error": None}


@router.post("/chat/follow-up", response_model=None)
async def chat_follow_up(request: FollowUpRequest) -> dict[str, Any] | JSONResponse:
    try:
        result = answer_follow_up(
            request.problem,
            request.remedy.model_dump(),
            request.question,
            [turn.model_dump() for turn in request.history],
        )
    except Exception as error:
        return _workflow_error(error)
    return {**result, "error": None}


@router.post("/chat", response_model=None)
async def chat(request: ChatRequest) -> dict[str, Any] | JSONResponse:
    question = request.question.strip()
    if not question:
        return error_response("Question must not be empty.", 422)
    if len(question) > MAX_QUESTION_LENGTH:
        return error_response(f"Question must be at most {MAX_QUESTION_LENGTH} characters.", 422)

    try:
        result = answer_question(question)
    except AnswerGenerationError as error:
        status_code = _error_status(error.category)
        logger.warning(
            "Chat request failed: category=%s status_code=%s",
            error.category,
            error.status_code,
        )
        return error_response(str(error), status_code)
    except Exception as error:
        logger.error("Unexpected chat request failure (%s)", type(error).__name__)
        return error_response("AyurBot could not process the question. Please try again.", 500)

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "error": None,
    }