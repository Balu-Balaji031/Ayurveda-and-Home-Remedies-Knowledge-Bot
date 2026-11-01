"""Offline tests for source-grounded remedy recommendations and follow-ups."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from google.genai import types

from app.rag import remedy_workflow, retriever
from app.rag.remedy_workflow import FALLBACK_LABEL, RemedyWorkflowError
from app.rag.retriever import RetrievedPassage, RetrievalResult


PASSAGE = RetrievedPassage(
    text="Ginger tea. Ingredients: fresh ginger and water. Simmer the slices in water and sip.",
    source="guide.pdf",
    page=4,
    chunk_id="guide-p0004-c0001",
)


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        text=PASSAGE.text,
        source=PASSAGE.source,
        page=PASSAGE.page,
        chunk_id=PASSAGE.chunk_id,
        distance=0.2,
        similarity=0.8,
    )


def run_tests() -> int:
    checks = 0
    extraction = {
        "mode": "pdf",
        "remedies": [{
            "name": "Ginger tea",
            "description": "Ginger tea.",
            "ingredients": "fresh ginger and water",
            "preparation": "Simmer the slices in water and sip.",
            "source_chunk_id": PASSAGE.chunk_id,
        }],
    }
    with patch.object(remedy_workflow, "retrieve_documents", return_value=[_retrieval_result()]):
        with patch.object(remedy_workflow, "generate_response", return_value=json.dumps(extraction)) as generate:
            response = remedy_workflow.recommend_remedies("cough")
    assert response["mode"] == "pdf"
    assert response["label"] is None
    assert len(response["remedies"]) == 1
    assert response["remedies"][0]["sources"] == [{
        "source": PASSAGE.source,
        "page": PASSAGE.page,
        "chunk_id": PASSAGE.chunk_id,
        "snippet": PASSAGE.text,
    }]
    assert generate.call_count == 1
    assert generate.call_args.kwargs == {
        "max_output_tokens": 2_048,
        "response_mime_type": "application/json",
        "response_json_schema": remedy_workflow.RECOMMENDATION_JSON_SCHEMA,
        "thinking_level": types.ThinkingLevel.MINIMAL,
    }
    checks += 1

    invalid_quote = {
        "mode": "pdf",
        "remedies": [{
            **extraction["remedies"][0],
            "description": "Ginger cures cough.",
        }],
    }
    fallback = {
        "mode": "ai_generated",
        "suggestions": [{
            "name": "Warm drink",
            "description": "Sip a comfortably warm, non-medicated drink.",
            "ingredients": None,
            "preparation": None,
            "traditional_or_supportive_basis": "Supportive comfort measure.",
            "evidence": "No disease-treatment evidence is claimed.",
            "possible_benefits": "May provide comfort.",
            "side_effects": "No specific risks are established by the available context.",
            "who_should_avoid": "Seek individual advice if unsure.",
            "precautions_interactions": "Do not use as a replacement for prescribed care.",
            "when_to_seek_care": "Seek care for severe or worsening symptoms.",
        }],
    }
    with patch.object(remedy_workflow, "retrieve_documents", return_value=[_retrieval_result()]):
        with patch.object(remedy_workflow, "generate_response", return_value=json.dumps(fallback)) as generate:
            response = remedy_workflow.recommend_remedies("cough")
    assert response["mode"] == "ai_generated"
    assert response["label"] == FALLBACK_LABEL
    assert response["sources"] == []
    assert response["remedies"][0]["kind"] == "ai_generated"
    assert generate.call_count == 1
    checks += 1

    with patch.object(remedy_workflow, "retrieve_documents", return_value=[]):
        with patch.object(
            remedy_workflow,
            "generate_response",
            return_value='{"mode":"ai_generated","suggestions":[]}',
        ):
            response = remedy_workflow.recommend_remedies("unclear symptoms")
    assert response["mode"] == "abstain"
    assert response["label"] is None
    assert response["remedies"] == []
    assert response["sources"] == []
    checks += 1

    with patch.object(remedy_workflow, "retrieve_documents", return_value=[_retrieval_result()]):
        with patch.object(remedy_workflow, "generate_response", return_value=json.dumps(invalid_quote)) as generate:
            response = remedy_workflow.recommend_remedies("cough")
    assert response["mode"] == "abstain"
    assert response["remedies"] == []
    assert generate.call_count == 1
    checks += 1

    pdf_remedy = {
        "id": "pdf:guide-p0004-c0001:1",
        "kind": "pdf",
        "name": "Ginger tea",
        "description": "Ginger tea.",
        "ingredients": "fresh ginger and water",
        "preparation": "Simmer the slices in water and sip.",
        "sources": [{
            "source": PASSAGE.source,
            "page": PASSAGE.page,
            "chunk_id": PASSAGE.chunk_id,
            "snippet": PASSAGE.text,
        }],
    }
    explanation = "**Remedy Name**\nGinger tea"
    with patch.object(remedy_workflow, "retrieve_documents_by_ids", return_value=[PASSAGE]) as get_sources:
        with patch.object(
            remedy_workflow,
            "generate_response",
            return_value=explanation,
        ) as generate:
            result = remedy_workflow.explain_remedy("cough", pdf_remedy)
    prompt = remedy_workflow._build_explanation_prompt("cough", pdf_remedy, [PASSAGE])
    assert get_sources.call_args.args[0] == [PASSAGE.chunk_id]
    assert "guide.pdf" in prompt and "page" in prompt
    assert "PDF passages as the primary reference" in prompt
    assert "Additional AI guidance (not stated in the PDF)" in prompt
    assert "Preparation – Step by Step" in prompt
    assert "11. Sources" not in prompt
    assert "Do not write a Source heading" in prompt
    assert "never add even a common household measure" in prompt
    assert "describe the preparation without measuring" in prompt
    assert result["answer"].startswith(explanation)
    assert "**Source**\n- PDF: guide.pdf, page 4" in result["answer"]
    assert "Additional AI guidance" in result["answer"]
    assert result["sources"][0]["source"] == PASSAGE.source
    assert generate.call_args.kwargs == {
        "max_output_tokens": 900,
        "thinking_level": types.ThinkingLevel.MINIMAL,
    }
    checks += 1

    incomplete_remedy = {
        **pdf_remedy,
        "name": "Warm water sips",
        "ingredients": None,
        "preparation": None,
        "sources": [{
            "source": PASSAGE.source,
            "page": 12,
            "chunk_id": "guide-p0012-c0002",
            "snippet": "Warm/hot water sips help to reduce dry cough.",
        }],
    }
    incomplete_passage = RetrievedPassage(
        text="Warm/hot water sips help to reduce dry cough.",
        source=PASSAGE.source,
        page=12,
        chunk_id="guide-p0012-c0002",
    )
    with patch.object(
        remedy_workflow,
        "retrieve_documents_by_ids",
        return_value=[incomplete_passage],
    ):
        with patch.object(remedy_workflow, "generate_response", return_value="Concise explanation.") as generate:
            remedy_workflow.explain_remedy("cough", incomplete_remedy)
    incomplete_prompt = generate.call_args.args[0]
    assert "may add cautious, well-established general guidance" in incomplete_prompt
    assert "never present it as a PDF fact" in incomplete_prompt
    assert "Only repeat a measurement, dose, frequency, or duration" in incomplete_prompt
    assert "describe the preparation without measuring" in incomplete_prompt
    assert "do not fill sections with generic insufficiency statements" in incomplete_prompt
    checks += 1

    with patch.object(remedy_workflow, "generate_response", return_value="Useful safe explanation."):
        ai_result = remedy_workflow.explain_remedy(
            "cough",
            {"id": "ai:warm-drink", "kind": "ai_generated", "name": "Warm drink"},
        )
    assert ai_result["sources"] == []
    assert ai_result["answer"].endswith(
        "**Source**\n- AI-generated suggestion – not found in uploaded PDFs."
    )
    checks += 1

    with patch.object(remedy_workflow, "retrieve_documents_by_ids", return_value=[PASSAGE]):
        with patch.object(remedy_workflow, "retrieve_documents", return_value=[_retrieval_result()]):
            with patch.object(remedy_workflow, "generate_response", return_value="Information is insufficient.") as generate:
                result = remedy_workflow.answer_follow_up(
                    "cough",
                    pdf_remedy,
                    "Can I use it while pregnant?",
                    [{"question": "What is this?", "answer": "Ginger tea."}],
                )
    follow_up_prompt = generate.call_args.args[0]
    assert "Can I use it while pregnant?" in follow_up_prompt
    assert "Ginger tea" in follow_up_prompt
    assert "What is this?" in follow_up_prompt
    assert "Answer only the user's follow-up directly" in follow_up_prompt
    assert generate.call_args.kwargs == {
        "max_output_tokens": 400,
        "thinking_level": types.ThinkingLevel.MINIMAL,
    }
    assert result["sources"][0]["chunk_id"] == PASSAGE.chunk_id
    checks += 1

    invalid_remedy = {**pdf_remedy, "sources": [{**pdf_remedy["sources"][0], "page": 99}]}
    with patch.object(remedy_workflow, "retrieve_documents_by_ids", return_value=[PASSAGE]):
        try:
            remedy_workflow.explain_remedy("cough", invalid_remedy)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("Mismatched PDF source metadata was accepted")

    with patch.object(remedy_workflow, "generate_response", return_value="not json"):
        try:
            remedy_workflow._json_response("test")
        except RemedyWorkflowError:
            checks += 1
        else:
            raise AssertionError("Malformed Gemini JSON was accepted")

    class FakeCollection:
        def get(self, *, ids, include):
            assert ids == [PASSAGE.chunk_id]
            assert include == ["documents", "metadatas"]
            return {
                "ids": [PASSAGE.chunk_id],
                "documents": [PASSAGE.text],
                "metadatas": [{
                    "source": PASSAGE.source,
                    "page": PASSAGE.page,
                    "chunk_id": PASSAGE.chunk_id,
                }],
            }

    with patch.object(retriever, "_open_existing_collection", return_value=FakeCollection()):
        assert retriever.retrieve_documents_by_ids([PASSAGE.chunk_id]) == [PASSAGE]
        checks += 1

    class MissingCollection:
        def get(self, *, ids, include):
            return {"ids": [], "documents": [], "metadatas": []}

    with patch.object(retriever, "_open_existing_collection", return_value=MissingCollection()):
        try:
            retriever.retrieve_documents_by_ids(["missing"])
        except retriever.RetrievalError:
            checks += 1
        else:
            raise AssertionError("A missing source passage ID was accepted")

    return checks


if __name__ == "__main__":
    print(f"Remedy-workflow tests: PASS ({run_tests()} checks)")
