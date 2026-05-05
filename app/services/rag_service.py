"""
rag_service.py
==============
Local RAG engine (no Qdrant needed).
Uses the rag_disease_db.json (structured disease data) and
rag_chunks/*.md (markdown documentation) for retrieval.

Flow:
  1. Given top-K predicted disease names → look up their entries in rag_disease_db
  2. Verify symptom overlap between user-checked symptoms and disease's core_symptoms
  3. Load the disease's markdown chunk file as document context
  4. Return structured context for LLM
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.constants import (
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_LENGTH,
    RAG_CHUNKS_DIR,
    RAG_DISEASE_DB_PATH,
)

# ─────────────────────────────────────────
# Load RAG database once
# ─────────────────────────────────────────

_RAG_DB: Optional[List[Dict]] = None
_RAG_DB_INDEX: Dict[str, Dict] = {}  # disease_name (lower) → entry


def _ensure_db_loaded():
    global _RAG_DB, _RAG_DB_INDEX
    if _RAG_DB is None:
        with open(RAG_DISEASE_DB_PATH, encoding="utf-8") as f:
            _RAG_DB = json.load(f)
        for entry in _RAG_DB:
            name = entry.get("disease_name", "")
            _RAG_DB_INDEX[name.lower()] = entry
        print(f"[RAG] Loaded {len(_RAG_DB)} disease entries.")


# ─────────────────────────────────────────
# Symptom matching
# ─────────────────────────────────────────


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def _symptom_overlap(
    user_symptoms: List[str], disease_symptoms: List[str]
) -> List[str]:
    """
    Find which user symptoms match disease's known symptoms.
    Uses word-level overlap for flexibility.
    """
    user_words = set()
    for s in user_symptoms:
        user_words.update(_normalize(s).split())

    matched = []
    for ds in disease_symptoms:
        ds_words = set(_normalize(ds).split())
        if ds_words & user_words:  # intersection
            matched.append(ds)

    return matched


def verify_symptoms_for_disease(
    disease_name: str, checked_symptoms: List[str]
) -> Tuple[bool, List[str]]:
    """
    Check if checked symptoms are consistent with this disease.
    Returns (is_match, matched_symptom_list).
    """
    _ensure_db_loaded()

    entry = _RAG_DB_INDEX.get(disease_name.lower())
    if not entry:
        return False, []

    core_syms = entry.get("core_symptoms", [])
    secondary_syms = entry.get("secondary_symptoms", [])
    all_disease_syms = core_syms + secondary_syms

    matched = _symptom_overlap(checked_symptoms, all_disease_syms)
    is_match = len(matched) > 0

    return is_match, matched


def candidate_diseases_for_symptoms(checked_symptoms: List[str]) -> Dict[str, List[str]]:
    """
    Build a candidate set of diseases that actually match the user's symptoms.
    Returns disease_name -> matched_symptoms.
    """
    _ensure_db_loaded()

    min_matches = 2 if len(checked_symptoms) >= 3 else 1
    candidates: Dict[str, List[str]] = {}

    for entry in _RAG_DB:
        disease_name = entry.get("disease_name", "")
        all_disease_symptoms = (
            entry.get("core_symptoms", [])
            + entry.get("secondary_symptoms", [])
            + entry.get("rare_symptoms", [])
        )
        matched = sorted(set(_symptom_overlap(checked_symptoms, all_disease_symptoms)))
        if len(matched) >= min_matches:
            candidates[disease_name] = matched

    return candidates


# ─────────────────────────────────────────
# Load markdown chunk for a disease
# ─────────────────────────────────────────


def _find_chunk_file(disease_name: str) -> Optional[Path]:
    """
    Find the .md chunk file for a disease.
    Tries multiple naming conventions.
    """
    chunks_dir = Path(RAG_CHUNKS_DIR)

    # Normalize disease name for filename matching
    clean = re.sub(r"['\s]+", "_", disease_name.strip())
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", clean)

    candidates = [
        chunks_dir / f"{clean}_chunk_1.md",
        chunks_dir / f"{clean}.md",
        chunks_dir / f"{clean}_chunk_2.md",
    ]

    # Also try lowercase
    for c in list(candidates):
        candidates.append(chunks_dir / c.name.lower())

    for path in candidates:
        if path.exists():
            return path

    # Fuzzy search: find any file whose stem contains a key word
    key_word = clean.split("_")[0].lower()
    for f in chunks_dir.glob("*.md"):
        if key_word in f.stem.lower():
            return f

    return None


def _load_chunk(disease_name: str) -> str:
    path = _find_chunk_file(disease_name)
    if path:
        return path.read_text(encoding="utf-8")
    return ""


# ─────────────────────────────────────────
# Build context for LLM
# ─────────────────────────────────────────


def build_rag_context(
    top_diseases: List[Dict], checked_symptoms: List[str]
) -> Tuple[str, List[Dict]]:
    """
    Given XGBoost top-K predictions and checked symptoms:
    1. Verify symptom match for each disease
    2. Load RAG chunk documentation
    3. Return (formatted_context_string, enriched_predictions_list)

    enriched_predictions_list: each dict has extra keys:
        rag_symptom_match: bool
        matched_symptoms: List[str]
    """
    _ensure_db_loaded()

    enriched = []
    context_blocks = []

    for pred in top_diseases:
        disease_name = pred["disease_name"]
        probability = pred["probability"]
        rank = pred["rank"]

        # Symptom verification
        is_match, matched = verify_symptoms_for_disease(disease_name, checked_symptoms)

        enriched.append(
            {
                **pred,
                "rag_symptom_match": is_match,
                "matched_symptoms": matched,
            }
        )

        # Load documentation chunk
        chunk_text = _load_chunk(disease_name)

        # Get structured entry from DB
        db_entry = _RAG_DB_INDEX.get(disease_name.lower(), {})

        # Format context block
        block_lines = [
            f"## Disease: {disease_name}",
            f"**Prediction Probability:** {probability:.1%} (Rank #{rank})",
            f"**Symptom Match:** {'✓ Verified' if is_match else '⚠ Partial match'}",
        ]

        if matched:
            block_lines.append(f"**Matched Symptoms:** {', '.join(matched[:5])}")

        if db_entry:
            severity = db_entry.get("severity", "")
            duration = db_entry.get("duration", "")
            if severity:
                block_lines.append(f"**Severity:** {severity}")
            if duration:
                block_lines.append(f"**Duration:** {duration}")

        if chunk_text:
            block_lines.append("")
            block_lines.append("### Medical Documentation:")
            block_lines.append(chunk_text[:2000])  # limit per disease

        context_blocks.append("\n".join(block_lines))

    # Join all disease blocks
    context = "\n\n---\n\n".join(context_blocks)

    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    return context, enriched


def get_disease_entry(disease_name: str) -> Optional[Dict]:
    """Get full RAG DB entry for a disease."""
    _ensure_db_loaded()
    return _RAG_DB_INDEX.get(disease_name.lower())


def get_all_disease_names() -> List[str]:
    _ensure_db_loaded()
    return [e.get("disease_name", "") for e in _RAG_DB]
