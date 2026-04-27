"""
xgboost_service.py
==================
Loads the pre-trained XGBoost model and label encoder.
Builds a one-row pandas DataFrame with the exact feature names expected by the
trained model, then returns top-K disease predictions.
"""

import json
import pickle
from typing import Dict, List

import pandas as pd
import xgboost as xgb

from core.constants import (
    FEATURE_DICT_PATH,
    LABEL_ENCODER_JSON_PATH,
    LABEL_ENCODER_PKL_PATH,
    MIN_PROBABILITY_THRESHOLD,
    SYMPTOM_VOCAB_PATH,
    TOP_K_DISEASES,
    XGBOOST_MODEL_PATH,
)


def _load_model() -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(str(XGBOOST_MODEL_PATH))
    return model


def _load_label_encoder():
    try:
        with open(LABEL_ENCODER_PKL_PATH, "rb") as handle:
            encoder = pickle.load(handle)
        return {"classes": list(encoder.classes_), "type": "sklearn"}
    except Exception:
        with open(LABEL_ENCODER_JSON_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            "classes": payload["classes"],
            "id_to_label": payload.get("id_to_label", {}),
            "type": "json",
        }


def _load_symptom_vocab() -> Dict[str, int]:
    with open(SYMPTOM_VOCAB_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _load_feature_dict() -> Dict[str, List[str]]:
    with open(FEATURE_DICT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


_MODEL = None
_LABEL_ENC = None
_SYMPTOM_VOCAB: Dict[str, int] = {}
_FEATURE_DICT: Dict[str, List[str]] = {}
_MODEL_FEATURE_NAMES: List[str] = []
_FEATURE_NAME_SET = set()


def _ensure_loaded():
    global _MODEL, _LABEL_ENC, _SYMPTOM_VOCAB, _FEATURE_DICT, _MODEL_FEATURE_NAMES, _FEATURE_NAME_SET

    if _MODEL is None:
        print("[XGBoost] Loading model...")
        _MODEL = _load_model()
        _LABEL_ENC = _load_label_encoder()
        _SYMPTOM_VOCAB = _load_symptom_vocab()
        _FEATURE_DICT = _load_feature_dict()
        _MODEL_FEATURE_NAMES = list(_MODEL.feature_names or [])
        _FEATURE_NAME_SET = set(_MODEL_FEATURE_NAMES)
        if not _MODEL_FEATURE_NAMES:
            raise RuntimeError("Loaded model has no feature names.")
        print(
            f"[XGBoost] Loaded. Features={len(_MODEL_FEATURE_NAMES)}, Classes={len(_LABEL_ENC['classes'])}"
        )


def _normalize_symptom(symptom: str) -> str:
    return symptom.lower().strip()


def _build_feature_frame(checked_symptoms: List[str]) -> pd.DataFrame:
    _ensure_loaded()

    row = {feature_name: 0.0 for feature_name in _MODEL_FEATURE_NAMES}
    matched = False

    for symptom in checked_symptoms:
        symptom_key = _normalize_symptom(symptom)
        if symptom_key in _FEATURE_NAME_SET:
            row[symptom_key] = 1.0
            matched = True

    if not matched:
        for symptom in checked_symptoms:
            for token in _normalize_symptom(symptom).split():
                if token in _FEATURE_NAME_SET:
                    row[token] = 1.0
                    matched = True

    return pd.DataFrame([row], columns=_MODEL_FEATURE_NAMES, dtype="float32")


def predict_diseases(checked_symptoms: List[str]) -> List[Dict]:
    _ensure_loaded()

    feature_frame = _build_feature_frame(checked_symptoms)
    dmatrix = xgb.DMatrix(feature_frame, feature_names=_MODEL_FEATURE_NAMES)
    probabilities = _MODEL.predict(dmatrix)[0]

    if not hasattr(probabilities, "__len__"):
        probabilities = [float(probabilities)]

    classes = _LABEL_ENC["classes"]
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)

    results = []
    for rank, (class_index, probability) in enumerate(ranked[:TOP_K_DISEASES], start=1):
        if probability < MIN_PROBABILITY_THRESHOLD:
            continue
        if class_index < len(classes):
            results.append(
                {
                    "disease_name": classes[class_index],
                    "probability": float(probability),
                    "rank": rank,
                }
            )

    return results


def get_all_symptoms() -> List[str]:
    _ensure_loaded()
    return list(_SYMPTOM_VOCAB.keys())


def get_symptom_vocab() -> Dict[str, int]:
    _ensure_loaded()
    return _SYMPTOM_VOCAB


def get_feature_dict() -> Dict[str, List[str]]:
    _ensure_loaded()
    return _FEATURE_DICT
