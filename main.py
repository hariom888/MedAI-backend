"""
main.py
=======
FastAPI application for the Medical Hybrid RAG-XGBoost Diagnostic Assistant.

Pipeline:
  POST /analyze-description  → LLM extracts symptom checkboxes from free text
  POST /generate-diagnosis   → XGBoost predicts + RAG verifies + LLM streams answer
  POST /followup             → LLM answers follow-up questions using RAG context
  GET  /health               → Service health check
"""

import json
import logging
from typing import AsyncGenerator, Optional

from api.dependencies import verify_api_key
from core.constants import GEMINI_MODELS
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from schemas.schemas import (
    DiseasePrediction,
    FollowUpRequest,
    HealthResponse,
    PredictionResult,
    SymptomCheckboxResponse,
    SymptomInputRequest,
    SymptomSelectionRequest,
)
from services import llm_service, rag_service, xgboost_service

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("medical_rag")

# ─────────────────────────────────────────
# App
# ─────────────────────────────────────────

app = FastAPI(
    title="Medical Hybrid RAG-XGBoost API",
    description=(
        "A diagnostic assistant combining XGBoost disease prediction, "
        "RAG-based medical documentation retrieval, and Gemini LLM generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Endpoint 1 — Symptom Extraction
# ─────────────────────────────────────────


@app.post(
    "/analyze-description",
    response_model=SymptomCheckboxResponse,
    summary="Extract symptom checkboxes from a free-text description",
    tags=["Diagnosis Pipeline"],
)
def analyze_description(
    body: SymptomInputRequest,
    _: str = Depends(verify_api_key),
    x_gemini_key: Optional[str] = Header(default=None, alias="X-Gemini-Key"),
) -> SymptomCheckboxResponse:
    """
    Step 1 of the pipeline.

    Accepts a free-text patient description and returns a structured list of
    symptom checkboxes for the user to confirm or adjust.
    """
    logger.info("analyze-description | description_len=%d", len(body.description))

    if not body.description.strip():
        raise HTTPException(status_code=400, detail="Description must not be empty.")

    try:
        result = llm_service.extract_symptom_checkboxes(
            description=body.description,
            history=[m.model_dump() for m in body.history],
            gemini_api_key=x_gemini_key,
        )
    except RuntimeError as exc:
        logger.error("LLM extraction failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}")

    # Validate that the LLM returned at least one checkbox
    checkboxes = result.get("checkboxes", [])
    if not checkboxes:
        raise HTTPException(
            status_code=500,
            detail="LLM returned no symptom checkboxes. Please try rephrasing your description.",
        )

    return SymptomCheckboxResponse(
        clarifying_message=result.get(
            "clarifying_message",
            "Please confirm which symptoms apply to you:",
        ),
        checkboxes=checkboxes,
        follow_up_question=result.get("follow_up_question"),
    )


# ─────────────────────────────────────────
# Endpoint 2 — Full Diagnosis (Streaming)
# ─────────────────────────────────────────


@app.post(
    "/generate-diagnosis",
    summary="Run the full XGBoost → RAG → LLM pipeline and stream the answer",
    tags=["Diagnosis Pipeline"],
)
def generate_diagnosis(
    body: SymptomSelectionRequest,
    _: str = Depends(verify_api_key),
    x_gemini_key: Optional[str] = Header(default=None, alias="X-Gemini-Key"),
) -> StreamingResponse:
    """
    Step 2 of the pipeline.

    1. XGBoost predicts the top diseases from checked symptoms.
    2. RAG verifies predictions against the disease database and loads documentation.
    3. Gemini streams a human-readable medical explanation grounded in the RAG context.

    **Response format** (newline-delimited text stream):

    - First chunk: `[PREDICTION_JSON]<json>\\n` — machine-readable predictions.
    - Subsequent chunks: plain text fragments of the LLM answer.
    """
    logger.info(
        "generate-diagnosis | symptoms=%d | model=%s",
        len(body.checked_symptoms),
        body.preferred_model or "default",
    )

    if not body.checked_symptoms:
        raise HTTPException(
            status_code=400,
            detail="At least one symptom must be checked before running diagnosis.",
        )

    # ── Step A: XGBoost prediction ──────────────────────────────────────────
    try:
        raw_predictions = xgboost_service.predict_diseases(body.checked_symptoms)
    except Exception as exc:
        logger.error("XGBoost prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction model error: {exc}")

    if not raw_predictions:
        # Graceful fallback: no disease cleared the probability threshold
        logger.warning("XGBoost returned no predictions above threshold.")
        raise HTTPException(
            status_code=422,
            detail=(
                "The model could not identify a likely disease from the provided symptoms. "
                "Please add more specific symptoms and try again."
            ),
        )

    # ── Step B: RAG context + symptom verification ───────────────────────────
    try:
        rag_context, enriched_predictions = rag_service.build_rag_context(
            top_diseases=raw_predictions,
            checked_symptoms=body.checked_symptoms,
        )
    except Exception as exc:
        logger.error("RAG context build failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"RAG service error: {exc}")

    # Build Pydantic-validated prediction payload
    disease_predictions = [
        DiseasePrediction(
            disease_name=p["disease_name"],
            probability=p["probability"],
            rank=p["rank"],
            rag_symptom_match=p.get("rag_symptom_match", False),
            matched_symptoms=p.get("matched_symptoms", []),
        )
        for p in enriched_predictions
    ]

    prediction_result = PredictionResult(
        top_diseases=disease_predictions,
        checked_symptoms=body.checked_symptoms,
    )

    # ── Step C: Streaming LLM response ──────────────────────────────────────
    async def stream_response() -> AsyncGenerator[str, None]:
        # Emit the prediction JSON as the very first chunk so the client
        # can render disease cards before the prose answer arrives.
        prediction_json = prediction_result.model_dump_json()
        yield f"[PREDICTION_JSON]{prediction_json}\n"

        try:
            for text_chunk in llm_service.generate_medical_answer(
                description=body.description,
                checked_symptoms=body.checked_symptoms,
                rag_context=rag_context,
                top_diseases=enriched_predictions,
                history=[m.model_dump() for m in body.history],
                model_name=body.preferred_model,
                gemini_api_key=x_gemini_key,
            ):
                yield text_chunk
        except Exception as exc:
            logger.error("LLM streaming error: %s", exc)
            yield f"\n\n⚠️ An error occurred while generating the medical answer: {exc}"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
        headers={
            # Allow the client to read the prediction payload from the header
            # as a convenience alternative to parsing the first stream chunk.
            "X-Prediction-Summary": json.dumps(
                [
                    {"disease": d.disease_name, "rank": d.rank}
                    for d in disease_predictions
                ]
            ),
            "X-Symptoms-Count": str(len(body.checked_symptoms)),
            "Cache-Control": "no-cache",
        },
    )


# ─────────────────────────────────────────
# Endpoint 3 — Follow-up Q&A (Streaming)
# ─────────────────────────────────────────


@app.post(
    "/followup",
    summary="Answer a follow-up question using the existing RAG context",
    tags=["Diagnosis Pipeline"],
)
def followup(
    body: FollowUpRequest,
    _: str = Depends(verify_api_key),
    x_gemini_key: Optional[str] = Header(default=None, alias="X-Gemini-Key"),
) -> StreamingResponse:
    """
    Allows the user to ask clarifying questions after receiving their diagnosis.

    RAG context is rebuilt for the diseases named in `context_diseases` so the
    LLM stays grounded in verified medical documentation.
    """
    logger.info(
        "followup | question_len=%d | context_diseases=%s",
        len(body.question),
        body.context_diseases,
    )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Re-fetch RAG context for the diseases from the prior diagnosis session
    rag_context = ""
    if body.context_diseases:
        try:
            stub_predictions = [
                {"disease_name": name, "probability": 1.0, "rank": i + 1}
                for i, name in enumerate(body.context_diseases)
            ]
            rag_context, _ = rag_service.build_rag_context(
                top_diseases=stub_predictions,
                checked_symptoms=[],  # no symptom re-verification needed
            )
        except Exception as exc:
            logger.warning("RAG context rebuild for follow-up failed: %s", exc)
            # Proceed with empty context rather than hard-failing

    async def stream_followup() -> AsyncGenerator[str, None]:
        try:
            for text_chunk in llm_service.generate_followup_answer(
                question=body.question,
                rag_context=rag_context,
                history=[m.model_dump() for m in body.history],
                context_diseases=body.context_diseases,
                model_name=body.preferred_model,
                gemini_api_key=x_gemini_key,
            ):
                yield text_chunk
        except Exception as exc:
            logger.error("Follow-up streaming error: %s", exc)
            yield f"\n\n⚠️ Error generating follow-up answer: {exc}"

    return StreamingResponse(
        stream_followup(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"},
    )


# ─────────────────────────────────────────
# Endpoint 4 — Health Check
# ─────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Utility"],
)
def health() -> HealthResponse:
    """Returns service status and the list of supported Gemini models."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_models=GEMINI_MODELS,
    )
