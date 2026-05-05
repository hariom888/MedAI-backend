from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# ----------------------------
# Chat / Conversation
# ----------------------------


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


# ----------------------------
# Step 1: Describe symptoms (free text)
# ----------------------------


class SymptomInputRequest(BaseModel):
    """User describes how they feel in natural language."""

    description: str  # "I have been feeling feverish for 3 days with headache"
    history: List[ChatMessage] = []


class SymptomCheckbox(BaseModel):
    """A single symptom checkbox item returned by LLM."""

    symptom_key: str  # matches symptom_vocab key (e.g. "fever")
    symptom_label: str  # human-readable label (e.g. "Fever / High temperature")
    checked: bool = False  # user will toggle this


class SymptomCheckboxResponse(BaseModel):
    """LLM returns a list of relevant symptom checkboxes."""

    clarifying_message: (
        str  # "Based on what you told me, please confirm which symptoms apply:"
    )
    checkboxes: List[SymptomCheckbox]
    follow_up_question: Optional[str] = None


# ----------------------------
# Step 2: User submits checked symptoms
# ----------------------------


class SymptomSelectionRequest(BaseModel):
    """User submits their checked symptoms."""

    checked_symptoms: List[str]  # list of symptom_key strings that were checked
    description: str  # original free-text description
    history: List[ChatMessage] = []
    preferred_model: Optional[str] = (
        None  # e.g. "gemini-2.0-flash" or "gemini-1.5-flash"
    )


# ----------------------------
# Step 3: XGBoost prediction result
# ----------------------------


class DiseasePrediction(BaseModel):
    disease_name: str
    probability: float
    display_confidence: float = 0.0
    rank: int
    rag_symptom_match: bool  # did RAG verify symptoms match this disease?
    matched_symptoms: List[str]  # symptoms that matched disease's known symptoms
    risk_level: str = "low"
    risk_score: float = 0.0
    probability_margin: float = 0.0
    matched_symptom_count: int = 0


class PredictionResult(BaseModel):
    top_diseases: List[DiseasePrediction]
    checked_symptoms: List[str]


# ----------------------------
# Step 4: Final medical answer
# ----------------------------


class DiagnosisRequest(BaseModel):
    """Full diagnosis flow triggered after symptom selection."""

    checked_symptoms: List[str]
    description: str
    history: List[ChatMessage] = []
    preferred_model: Optional[str] = None


class DiagnosisResponse(BaseModel):
    predictions: PredictionResult
    medical_answer: str  # streamed separately or returned here
    model_used: str


# ----------------------------
# Follow-up Q&A
# ----------------------------


class FollowUpRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    context_diseases: List[str] = []  # disease names from last prediction
    preferred_model: Optional[str] = None


# ----------------------------
# Session snapshot (saved to JSON)
# ----------------------------


class SessionSnapshot(BaseModel):
    session_id: str
    description: str
    checked_symptoms: List[str]
    predictions: List[DiseasePrediction]
    timestamp: str


# ----------------------------
# Misc
# ----------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    available_models: List[str]
