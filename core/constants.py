import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAG_CHUNKS_DIR = BASE_DIR / "rag_chunks"


def _resolve_data_file(filename: str) -> Path:
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path

    root_path = BASE_DIR / filename
    if root_path.exists():
        return root_path

    return data_path

# XGBoost model paths
XGBOOST_MODEL_PATH = _resolve_data_file("medical_model.xgb")
LABEL_ENCODER_PKL_PATH = _resolve_data_file("label_encoder.pkl")
LABEL_ENCODER_JSON_PATH = _resolve_data_file("label_encoder.json")
SYMPTOM_VOCAB_PATH = _resolve_data_file("symptom_vocab.json")
FEATURE_DICT_PATH = _resolve_data_file("feature_dictionary.json")
RAG_DISEASE_DB_PATH = _resolve_data_file("rag_disease_db.json")

# XGBoost prediction
TOP_K_DISEASES = 3
MIN_PROBABILITY_THRESHOLD = 0.01

# RAG retrieval
MAX_CONTEXT_CHUNKS = 4
MAX_CONTEXT_LENGTH = 8000
SCORE_THRESHOLD = 0.3

# Gemini free tier models (in order of preference)
GEMINI_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]
