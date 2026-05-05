"""
Loads the trained disease model and returns calibrated top-k predictions.
"""

from __future__ import annotations

import json
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from core.constants import (
    CALIBRATED_MODEL_PATH,
    FEATURE_DICT_PATH,
    LABEL_ENCODER_JSON_PATH,
    LABEL_ENCODER_PKL_PATH,
    MIN_PROBABILITY_THRESHOLD,
    MODEL_METADATA_PATH,
    SYMPTOM_VOCAB_PATH,
    TOP_K_DISEASES,
    XGBOOST_MODEL_PATH,
)
from services import rag_service


def _load_calibrated_model():
    if CALIBRATED_MODEL_PATH.exists():
        with open(CALIBRATED_MODEL_PATH, "rb") as handle:
            return pickle.load(handle)
    return None


def _load_raw_model() -> xgb.Booster:
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
    try:
        with open(FEATURE_DICT_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


def _load_model_metadata() -> dict:
    try:
        with open(MODEL_METADATA_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {"symptom_columns": [], "interaction_pairs": [], "rarity_weights": {}}


_CALIBRATED_MODEL = None
_RAW_MODEL = None
_LABEL_ENC = None
_SYMPTOM_VOCAB: Dict[str, int] = {}
_FEATURE_DICT: Dict[str, List[str]] = {}
_MODEL_METADATA: dict = {}
_MODEL_FEATURE_NAMES: List[str] = []
_FEATURE_NAME_SET = set()
_LOW_CONFIDENCE_TOP_PROBABILITY = 0.15
_DISPLAY_CONFIDENCE_ANCHORS = [
    (0.00, 0.0),
    (0.10, 12.0),
    (0.15, 20.0),
    (0.25, 40.0),
    (0.30, 50.0),
    (0.50, 70.0),
    (0.70, 82.0),
    (0.90, 89.0),
    (1.00, 89.0),
]


def _ensure_loaded():
    global _CALIBRATED_MODEL, _RAW_MODEL, _LABEL_ENC, _SYMPTOM_VOCAB
    global _FEATURE_DICT, _MODEL_METADATA, _MODEL_FEATURE_NAMES, _FEATURE_NAME_SET

    if _RAW_MODEL is None:
        print("[XGBoost] Loading model...")
        _CALIBRATED_MODEL = _load_calibrated_model()
        _RAW_MODEL = _load_raw_model()
        _LABEL_ENC = _load_label_encoder()
        _SYMPTOM_VOCAB = _load_symptom_vocab()
        _FEATURE_DICT = _load_feature_dict()
        _MODEL_METADATA = _load_model_metadata()

        if _CALIBRATED_MODEL is not None and hasattr(_CALIBRATED_MODEL, "feature_names_in_"):
            _MODEL_FEATURE_NAMES = list(_CALIBRATED_MODEL.feature_names_in_)
        else:
            _MODEL_FEATURE_NAMES = list(_RAW_MODEL.feature_names or [])

        _FEATURE_NAME_SET = set(_MODEL_FEATURE_NAMES)
        if not _MODEL_FEATURE_NAMES:
            raise RuntimeError("Loaded model has no feature names.")

        print(
            f"[XGBoost] Loaded. Features={len(_MODEL_FEATURE_NAMES)}, "
            f"Classes={len(_LABEL_ENC['classes'])}, Calibrated={_CALIBRATED_MODEL is not None}"
        )


def _normalize_symptom(symptom: str) -> str:
    return symptom.lower().strip().replace(" ", "_")


def _extract_checked_symptom_set(checked_symptoms: List[str]) -> set[str]:
    _ensure_loaded()
    matched = set()

    for symptom in checked_symptoms:
        normalized = _normalize_symptom(symptom)
        if normalized in _FEATURE_NAME_SET:
            matched.add(normalized)
            continue

        for token in normalized.split("_"):
            if token in _FEATURE_NAME_SET:
                matched.add(token)

    return matched


def _build_feature_frame(checked_symptoms: List[str]) -> pd.DataFrame:
    _ensure_loaded()

    row = {feature_name: 0.0 for feature_name in _MODEL_FEATURE_NAMES}
    active_symptoms = _extract_checked_symptom_set(checked_symptoms)
    metadata_symptoms = _MODEL_METADATA.get("symptom_columns", [])

    target_symptoms = metadata_symptoms or [name for name in _MODEL_FEATURE_NAMES if not name.startswith("pair__")]

    for symptom_name in target_symptoms:
        if symptom_name in active_symptoms and symptom_name in row:
            row[symptom_name] = 1.0

    if "symptom_count" in row:
        row["symptom_count"] = float(
            sum(1 for symptom_name in target_symptoms if symptom_name in active_symptoms)
        )

    rarity_score = 0.0
    for symptom_name, weight in _MODEL_METADATA.get("rarity_weights", {}).items():
        if symptom_name in active_symptoms:
            rarity_score += float(weight)
    if "rarity_score" in row:
        row["rarity_score"] = rarity_score

    for left, right in _MODEL_METADATA.get("interaction_pairs", []):
        feature_name = f"pair__{left}__{right}"
        if feature_name in row and left in active_symptoms and right in active_symptoms:
            row[feature_name] = 1.0

    return pd.DataFrame([row], columns=_MODEL_FEATURE_NAMES, dtype="float32")


def _predict_probabilities(feature_frame: pd.DataFrame):
    if _CALIBRATED_MODEL is not None:
        probabilities = _CALIBRATED_MODEL.predict_proba(feature_frame)[0]
    else:
        dmatrix = xgb.DMatrix(feature_frame, feature_names=_MODEL_FEATURE_NAMES)
        probabilities = _RAW_MODEL.predict(dmatrix)[0]

    if not hasattr(probabilities, "__len__"):
        probabilities = [float(probabilities)]
    return probabilities


def _restrict_to_symptom_matched_diseases(probabilities, checked_symptoms: List[str]):
    candidates = rag_service.candidate_diseases_for_symptoms(checked_symptoms)
    classes = _LABEL_ENC["classes"]

    if not candidates:
        return np.asarray(probabilities, dtype=np.float64), {}

    filtered = np.zeros(len(classes), dtype=np.float64)
    matched_lookup: Dict[str, List[str]] = {}

    for idx, class_name in enumerate(classes):
        if class_name in candidates:
            filtered[idx] = float(probabilities[idx])
            matched_lookup[class_name] = candidates[class_name]

    total = float(filtered.sum())
    if total <= 0:
        return np.asarray(probabilities, dtype=np.float64), {}

    filtered /= total
    return filtered, matched_lookup


def _compute_risk_annotations(
    ranked_probabilities: List[tuple[int, float]],
    classes: List[str],
    matched_lookup: Dict[str, List[str]],
):
    annotations: Dict[int, Dict[str, object]] = {}

    for position, (class_index, probability) in enumerate(ranked_probabilities):
        next_probability = (
            float(ranked_probabilities[position + 1][1])
            if position + 1 < len(ranked_probabilities)
            else 0.0
        )
        margin = max(float(probability) - next_probability, 0.0)
        match_count = len(matched_lookup.get(classes[class_index], []))
        evidence_score = min(match_count / 4.0, 1.0)

        base_score = (
            0.60 * float(probability)
            + 0.25 * margin
            + 0.15 * evidence_score
        )
        rank_penalty = 0.18 * position
        risk_score = max(min(base_score - rank_penalty, 1.0), 0.0)

        if risk_score >= 0.75 and float(probability) >= 0.70 and margin >= 0.20:
            risk_level = "high"
        elif risk_score >= 0.42 and float(probability) >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        annotations[class_index] = {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 4),
            "probability_margin": round(margin, 4),
            "matched_symptom_count": match_count,
        }

    return annotations


def _interpolate_display_confidence(probability: float) -> float:
    probability = max(0.0, min(float(probability), 1.0))

    for idx in range(len(_DISPLAY_CONFIDENCE_ANCHORS) - 1):
        left_p, left_score = _DISPLAY_CONFIDENCE_ANCHORS[idx]
        right_p, right_score = _DISPLAY_CONFIDENCE_ANCHORS[idx + 1]
        if left_p <= probability <= right_p:
            if right_p == left_p:
                return round(left_score, 1)
            ratio = (probability - left_p) / (right_p - left_p)
            return round(left_score + ratio * (right_score - left_score), 1)

    return 89.0


def predict_diseases(checked_symptoms: List[str]) -> List[Dict]:
    _ensure_loaded()

    feature_frame = _build_feature_frame(checked_symptoms)
    probabilities = _predict_probabilities(feature_frame)
    probabilities, matched_lookup = _restrict_to_symptom_matched_diseases(
        probabilities, checked_symptoms
    )
    classes = _LABEL_ENC["classes"]
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    annotations = _compute_risk_annotations(ranked, classes, matched_lookup)

    if not ranked or float(ranked[0][1]) < _LOW_CONFIDENCE_TOP_PROBABILITY:
        return []

    results = []
    for rank, (class_index, probability) in enumerate(ranked[:TOP_K_DISEASES], start=1):
        if probability < MIN_PROBABILITY_THRESHOLD:
            continue
        if class_index < len(classes):
            risk_annotation = annotations.get(class_index, {})
            results.append(
                {
                    "disease_name": classes[class_index],
                    "probability": float(probability),
                    "display_confidence": _interpolate_display_confidence(float(probability)),
                    "rank": rank,
                    "candidate_symptom_matches": matched_lookup.get(classes[class_index], []),
                    "risk_level": risk_annotation.get("risk_level", "low"),
                    "risk_score": risk_annotation.get("risk_score", 0.0),
                    "probability_margin": risk_annotation.get("probability_margin", 0.0),
                    "matched_symptom_count": risk_annotation.get("matched_symptom_count", 0),
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
