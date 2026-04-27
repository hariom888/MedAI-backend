#!/usr/bin/env python3
"""
Medical Dataset Pipeline
========================
Steps:
  1. Parse disease_templates.json → clean RAG JSON per disease
  2. Chunk RAG JSON into Markdown blocks for vector embedding
  3. Generate XGBoost training CSV with 250 samples/disease,
     probabilistic symptom dropout (15-20%), and noise injection (1%)

Usage:
  python medical_rag_xgboost_pipeline.py

Outputs:
  rag_disease_db.json          – Structured RAG database (one entry per disease)
  rag_chunks/                  – Folder of Markdown .md chunk files for embedding
  xgboost_training_data.csv   – Tabular training data for XGBoost
"""

import json
import os
import re
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from excluded_diseases import is_excluded_disease

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
TEMPLATES_FILE    = "disease_templates.json"
VOCAB_FILE        = "symptom_vocab.json"
FEATURE_DICT_FILE = "feature_dictionary.json"
LABEL_ENC_FILE    = "label_encoder.json"
CLASS_DIST_FILE   = "class_distribution.json"
TRAIN_CSV         = "train.csv"

RAG_DB_OUT        = "rag_disease_db.json"
RAG_CHUNKS_DIR    = "rag_chunks"
XGBOOST_CSV_OUT   = "xgboost_training_data.csv"

SAMPLES_PER_DISEASE = 250
DROPOUT_MIN         = 0.15   # 15% chance a core symptom is missing
DROPOUT_MAX         = 0.20   # 20% upper bound
NOISE_PROB          = 0.01   # 1% chance patient has a random unrelated symptom
RANDOM_SEED         = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════
# STEP 1 – DATA ANALYSIS & SCHEMA DESIGN
# Build a clean, enriched RAG JSON database from source templates.
# ══════════════════════════════════════════════════════════════════

def load_source_data():
    """Load all source JSON files."""
    with open(TEMPLATES_FILE, encoding="utf-8") as f:
        templates = json.load(f)
    with open(VOCAB_FILE, encoding="utf-8") as f:
        _vocab_raw = json.load(f)
        # symptom_vocab.json is a word→index dict; extract the word set as vocab
        vocab = set(_vocab_raw.keys())
    with open(FEATURE_DICT_FILE, encoding="utf-8") as f:
        feature_dict = json.load(f)
    with open(LABEL_ENC_FILE, encoding="utf-8") as f:
        label_enc = json.load(f)
    with open(CLASS_DIST_FILE, encoding="utf-8") as f:
        class_dist = json.load(f)
    return templates, vocab, feature_dict, label_enc, class_dist


def filter_templates(templates):
    """Drop excluded diseases from the template dictionary."""
    if isinstance(templates, dict):
        return {
            name: data for name, data in templates.items()
            if not is_excluded_disease(name)
        }
    return [item for item in templates if not is_excluded_disease(item.get("name", ""))]


def humanize(snake: str) -> str:
    """Convert snake_case symptom column name to readable text."""
    return snake.replace("_", " ").strip()


def compute_symptom_prevalence_from_train(symptom_cols: list[str]) -> dict[str, dict[str, float]]:
    """
    For every disease in train.csv, compute the fraction of rows where
    each symptom is 1.  This gives us data-driven 'core' vs 'secondary'
    thresholds instead of guessing.
    Returns: { disease_name: { symptom_col: prevalence_fraction } }

    train.csv stores symptoms as a packed comma-separated binary string in
    the 'symptom_vector' column, indexed by symptom_vocab.json (word→index).
    We expand this vector and map position→slug to compute prevalences.
    """
    print("  Computing symptom prevalence from train.csv (this may take a moment)…")

    # Build index→slug mapping from vocab + template slugs
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab_raw = json.load(f)   # word → index

    # Read train.csv
    df = pd.read_csv(TRAIN_CSV, low_memory=True)
    df = df[df["disease"].notna() & df["symptom_vector"].notna()]
    df = df[~df["disease"].astype(str).map(is_excluded_disease)]

    if df.empty or "symptom_vector" not in df.columns:
        print("  [WARN] train.csv has no usable symptom_vector column; skipping prevalence.")
        return {}

    # Expand symptom_vector into a matrix
    vec_matrix = df["symptom_vector"].apply(
        lambda s: [int(x) for x in s.split(",")]
    )
    vec_len = vec_matrix.iloc[0].__len__() if len(vec_matrix) else 0

    # Build position→symptom_col mapping using vocab index
    # vocab_raw is word→index; we need index→word, then word→slug
    index_to_slug: dict[int, str] = {}
    for word, idx in vocab_raw.items():
        if isinstance(idx, int) and idx < vec_len:
            slug = re.sub(r"[^a-z0-9]+", "_", word.lower()).strip("_")
            if slug in set(symptom_cols):
                index_to_slug[idx] = slug

    # Convert matrix to DataFrame with slug column names
    positions = sorted(index_to_slug.keys())
    if not positions:
        print("  [WARN] No vocab→column matches found; skipping prevalence.")
        return {}

    expanded = pd.DataFrame(
        [[row[i] for i in positions] for row in vec_matrix],
        columns=[index_to_slug[i] for i in positions],
        index=df.index,
    )
    expanded["disease"] = df["disease"].values

    symptom_col_set = set(symptom_cols)
    prevalence: dict[str, dict[str, float]] = {}
    for disease, group in expanded.groupby("disease"):
        if len(group) == 0:
            continue
        avail_cols = [c for c in group.columns if c != "disease" and c in symptom_col_set]
        means = group[avail_cols].mean()
        prevalence[str(disease)] = {col: float(v) for col, v in means.items() if v > 0}

    print(f"  Prevalence computed for {len(prevalence)} diseases.")
    return prevalence


def build_rag_database(templates, vocab, prevalence, label_enc) -> list[dict]:
    """
    Build enriched RAG records.

    Schema per record:
      disease_name     str        – canonical disease name
      disease_id       int        – integer label used in XGBoost
      aliases          list[str]  – common alternative names
      icd_hint         str        – rough ICD-10 chapter (heuristic)
      severity         str        – mild / moderate / severe / critical
      duration         str        – acute / subacute / chronic
      onset_speed      str        – sudden / gradual / insidious
      core_symptoms    list[str]  – prevalence ≥ 40% in training data
      secondary_symptoms list[str]– prevalence 10-39% in training data
      rare_symptoms    list[str]  – prevalence < 10% but > 0%
      emergency_signs  list[str]  – from template
      risk_groups      list[str]  – from template
      complications    list[str]  – from template
      all_symptoms_raw list[str]  – unfiltered from template (for XGBoost dropout)
      symptom_count    int        – total unique symptoms
      semantic_summary str        – pre-built natural language description for RAG
      chunk_overlap_keys list[str]– key terms to repeat in overlapping chunks
      created_at       str        – ISO timestamp
    """
    label_to_int = label_enc.get("label_to_int", {})
    rag_db = []

    # disease_templates.json is a dict {disease_name: data}, not a list
    template_items = templates.items() if isinstance(templates, dict) else [(t["name"], t) for t in templates]
    for name, t in template_items:
        raw_symptoms = [s for s in t.get("symptoms", []) if s in vocab or True]
        # Remove blank/too-short entries
        raw_symptoms = [s for s in raw_symptoms if len(s) > 2]

        # Prevalence-based tiering from train.csv
        prev = prevalence.get(name, {})
        core_symptoms      = [s for s, v in prev.items() if v >= 0.40]
        secondary_symptoms = [s for s, v in prev.items() if 0.10 <= v < 0.40]
        rare_symptoms      = [s for s, v in prev.items() if 0 < v < 0.10]

        # Fallback: if disease not in train.csv (new disease) use template list directly
        if not core_symptoms and raw_symptoms:
            core_symptoms = raw_symptoms[:max(1, len(raw_symptoms) // 2)]
            secondary_symptoms = raw_symptoms[len(core_symptoms):]

        # Humanize for RAG
        def h_list(lst): return [humanize(s) for s in lst]

        # Build semantic summary for the chunk
        summary_parts = [
            f"{name} is a {t.get('severity','unknown severity')} condition "
            f"with a {t.get('duration','variable')} course."
        ]
        if core_symptoms:
            summary_parts.append(
                f"Core symptoms include: {', '.join(h_list(core_symptoms[:8]))}."
            )
        if secondary_symptoms:
            summary_parts.append(
                f"Secondary symptoms may include: {', '.join(h_list(secondary_symptoms[:6]))}."
            )
        if t.get("risk_groups"):
            summary_parts.append(
                f"Risk groups: {', '.join(t['risk_groups'][:5])}."
            )
        if t.get("emergency_signs"):
            summary_parts.append(
                f"Emergency warning signs: {', '.join(h_list(t['emergency_signs'][:5]))}."
            )
        semantic_summary = " ".join(summary_parts)

        record = {
            "disease_name":        name,
            "disease_id":          label_to_int.get(name, -1),
            "aliases":             [],           # can be populated later from SNOMED/ICD
            "severity":            t.get("severity", "unknown"),
            "duration":            t.get("duration", "unknown"),
            "onset_speed":         "unknown",    # not in template; derive from train stats
            "core_symptoms":       core_symptoms,
            "secondary_symptoms":  secondary_symptoms,
            "rare_symptoms":       rare_symptoms,
            "emergency_signs":     t.get("emergency_signs", []),
            "risk_groups":         t.get("risk_groups", []),
            "complications":       t.get("complications", []),
            "all_symptoms_raw":    raw_symptoms,
            "symptom_count":       len(set(core_symptoms + secondary_symptoms + rare_symptoms)),
            "semantic_summary":    semantic_summary,
            "chunk_overlap_keys":  [name] + h_list(core_symptoms[:3]),
            "created_at":          datetime.utcnow().isoformat(),
        }
        rag_db.append(record)

    print(f"  RAG database built: {len(rag_db)} disease records.")
    return rag_db


# ══════════════════════════════════════════════════════════════════
# STEP 2 – RAG OPTIMISATION: chunk into overlapping Markdown blocks
# ══════════════════════════════════════════════════════════════════

CHUNK_OVERLAP_SENTENCES = 2   # sentences to repeat at start of next chunk


def build_markdown_chunk(record: dict) -> str:
    """
    Generate a single comprehensive Markdown chunk per disease.
    Designed to be embedded as one atomic unit; long diseases can be
    split further with the multi_chunk function below.
    """
    def h(lst): return ", ".join(humanize(s) for s in lst) if lst else "None documented"

    lines = [
        f"# {record['disease_name']}",
        "",
        f"**Severity:** {record['severity'].capitalize()}  ",
        f"**Duration:** {record['duration'].capitalize()}  ",
        f"**Disease ID (XGBoost label):** {record['disease_id']}",
        "",
        "## Summary",
        record["semantic_summary"],
        "",
        "## Core Symptoms (prevalence ≥ 40%)",
        h(record["core_symptoms"]),
        "",
        "## Secondary Symptoms (prevalence 10–39%)",
        h(record["secondary_symptoms"]),
        "",
        "## Rare / Atypical Symptoms (prevalence < 10%)",
        h(record["rare_symptoms"]),
        "",
        "## Emergency / Red-Flag Signs",
        h(record["emergency_signs"]),
        "",
        "## Risk Groups",
        ", ".join(record["risk_groups"]) if record["risk_groups"] else "General population",
        "",
        "## Complications",
        ", ".join(record["complications"]) if record["complications"] else "None documented",
        "",
        "---",
        f"*Keywords for retrieval: {', '.join(record['chunk_overlap_keys'])}*",
    ]
    return "\n".join(lines)


def build_overlapping_chunks(record: dict) -> list[dict]:
    """
    For diseases with many symptoms, produce 2 overlapping chunks:
      Chunk A – identity + core symptoms + emergency signs
      Chunk B – core symptom overlap + secondary + rare + risk/complications
    The overlap_keys repeat in both chunks to aid vector retrieval.
    """
    name = record["disease_name"]
    overlap_intro = (
        f"[Context: This chunk continues the entry for **{name}**. "
        f"Core symptoms: {', '.join(humanize(s) for s in record['core_symptoms'][:3])}.]"
    )

    def h(lst): return ", ".join(humanize(s) for s in lst) if lst else "None documented"

    chunk_a = "\n".join([
        f"# {name} — Part 1: Core Clinical Profile",
        "",
        f"**Severity:** {record['severity'].capitalize()}  ",
        f"**Duration:** {record['duration'].capitalize()}",
        "",
        "## Summary",
        record["semantic_summary"],
        "",
        "## Core Symptoms (prevalence ≥ 40%)",
        h(record["core_symptoms"]),
        "",
        "## Emergency / Red-Flag Signs",
        h(record["emergency_signs"]),
        "",
        f"*Keywords: {', '.join(record['chunk_overlap_keys'])}*",
    ])

    chunk_b = "\n".join([
        f"# {name} — Part 2: Secondary Profile & Risk",
        "",
        overlap_intro,
        "",
        "## Secondary Symptoms (prevalence 10–39%)",
        h(record["secondary_symptoms"]),
        "",
        "## Rare / Atypical Symptoms",
        h(record["rare_symptoms"]),
        "",
        "## Risk Groups",
        ", ".join(record["risk_groups"]) if record["risk_groups"] else "General population",
        "",
        "## Complications",
        ", ".join(record["complications"]) if record["complications"] else "None documented",
        "",
        f"*Keywords: {', '.join(record['chunk_overlap_keys'])}*",
    ])

    return [
        {"chunk_id": f"{name}_chunk_1", "disease": name, "content": chunk_a},
        {"chunk_id": f"{name}_chunk_2", "disease": name, "content": chunk_b},
    ]


def write_rag_chunks(rag_db: list[dict], out_dir: str):
    """Write Markdown chunk files to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    all_chunks = []
    for record in rag_db:
        # Use 2-chunk strategy for diseases with ≥ 8 combined symptoms
        total_syms = len(record["core_symptoms"]) + len(record["secondary_symptoms"]) + len(record["rare_symptoms"])
        if total_syms >= 8:
            chunks = build_overlapping_chunks(record)
        else:
            single_md = build_markdown_chunk(record)
            safe_name = re.sub(r"[^\w\-]", "_", record["disease_name"])
            chunks = [{"chunk_id": f"{safe_name}_chunk_1",
                       "disease": record["disease_name"],
                       "content": single_md}]

        for chunk in chunks:
            safe_id = re.sub(r"[^\w\-]", "_", chunk["chunk_id"])
            filepath = os.path.join(out_dir, f"{safe_id}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(chunk["content"])
            all_chunks.append(chunk)

    print(f"  Written {len(all_chunks)} Markdown chunk files to '{out_dir}/'.")
    return all_chunks


# ══════════════════════════════════════════════════════════════════
# STEP 3 – XGBOOST DATA GENERATOR
# ══════════════════════════════════════════════════════════════════

def get_symptom_columns_from_train() -> list[str]:
    """Derive symptom binary column names from disease_templates.json.

    train.csv stores symptoms as a packed 'symptom_vector' string, not
    individual binary columns.  The actual binary columns used in the
    XGBoost dataset are built by comma-splitting each symptom phrase in
    the templates and slugifying each part — exactly how xgboost_training_data.csv
    was originally generated.  We reconstruct that set here and, if an
    existing xgboost CSV is present, validate against its columns.
    """
    with open(TEMPLATES_FILE, encoding="utf-8") as f:
        templates = json.load(f)

    # Build the full slug set from all template symptom phrases
    slug_set: set[str] = set()
    template_items = templates.items() if isinstance(templates, dict) else [(t["name"], t) for t in templates]
    for _name, data in template_items:
        for phrase in data.get("symptoms", []):
            for part in phrase.split(","):
                slug = re.sub(r"[^a-z0-9]+", "_", part.strip().lower()).strip("_")
                if slug:
                    slug_set.add(slug)

    # If an existing XGBoost CSV exists, use its column order (keeps reproducibility)
    if os.path.exists(XGBOOST_CSV_OUT):
        existing_cols = pd.read_csv(XGBOOST_CSV_OUT, nrows=0).columns.tolist()
        existing_symptom_cols = [c for c in existing_cols if c != "label"]
        # Union: keep existing order, append any new slugs not yet present
        extra = [s for s in sorted(slug_set) if s not in set(existing_symptom_cols)]
        return existing_symptom_cols + extra

    return sorted(slug_set)


def get_disease_symptom_map(rag_db: list[dict], symptom_cols: list[str]) -> dict[str, dict]:
    """
    For each disease return:
      core      – symptom cols that are core (high dropout risk)
      secondary – symptom cols that are secondary (higher dropout risk)
    Only include symptoms that are actual binary columns in the dataset.
    """
    col_set = set(symptom_cols)
    mapping = {}
    for record in rag_db:
        disease = record["disease_name"]
        core      = [s for s in record["core_symptoms"]      if s in col_set]
        secondary = [s for s in record["secondary_symptoms"] if s in col_set]
        # Fall back: if prevalence analysis returned nothing, use raw template symptoms
        if not core and not secondary:
            raw = [s for s in record["all_symptoms_raw"] if s in col_set]
            core      = raw[:max(1, len(raw) // 2)]
            secondary = raw[len(core):]
        mapping[disease] = {"core": core, "secondary": secondary}
    return mapping


def generate_patient_row(
    disease: str,
    disease_symptom_map: dict,
    all_symptom_cols: list[str],
    noise_prob: float = NOISE_PROB,
    dropout_min: float = DROPOUT_MIN,
    dropout_max: float = DROPOUT_MAX,
) -> dict:
    """
    Generate a single synthetic patient row for `disease`.

    Logic:
      1. Start with all symptom columns = 0.
      2. For each CORE symptom: assign 1, then apply probabilistic dropout
         (15–20% chance of flipping back to 0 — simulating missed/absent symptoms).
      3. For each SECONDARY symptom: assign 1 with 60% probability (they're not
         always present), then apply the same dropout.
      4. Noise injection: each non-disease symptom has NOISE_PROB chance of being 1.
    """
    row = {col: 0 for col in all_symptom_cols}

    core_syms = disease_symptom_map[disease]["core"]
    sec_syms  = disease_symptom_map[disease]["secondary"]
    dropout_rate = random.uniform(dropout_min, dropout_max)

    # Core symptoms: always start as 1, then apply dropout
    for sym in core_syms:
        if sym in row:
            row[sym] = 0 if random.random() < dropout_rate else 1

    # Secondary symptoms: present ~60% of the time, then apply dropout
    for sym in sec_syms:
        if sym in row:
            if random.random() < 0.60:
                row[sym] = 0 if random.random() < dropout_rate else 1

    # Noise: 1% chance of spurious symptom
    for col in all_symptom_cols:
        if row[col] == 0 and random.random() < noise_prob:
            row[col] = 1

    row["label"] = disease
    return row


def generate_xgboost_dataset(
    rag_db: list[dict],
    symptom_cols: list[str],
    disease_symptom_map: dict,
    samples_per_disease: int = SAMPLES_PER_DISEASE,
) -> pd.DataFrame:
    """
    Generate the full tabular dataset:
      - 250 rows per disease
      - Columns: all symptom binary features + 'label'
    """
    print(f"  Generating {samples_per_disease} samples × {len(rag_db)} diseases "
          f"= {samples_per_disease * len(rag_db):,} total rows…")

    rows = []
    disease_names = [r["disease_name"] for r in rag_db]

    for i, disease in enumerate(disease_names):
        if disease not in disease_symptom_map:
            # Disease has no symptom mapping – skip with warning
            print(f"  [WARN] No symptom mapping for '{disease}', skipping.")
            continue
        for _ in range(samples_per_disease):
            row = generate_patient_row(disease, disease_symptom_map, symptom_cols)
            rows.append(row)

        if (i + 1) % 25 == 0:
            print(f"    Processed {i + 1}/{len(disease_names)} diseases…")

    df = pd.DataFrame(rows)
    # Ensure column order: symptoms first, label last
    ordered_cols = symptom_cols + ["label"]
    df = df[ordered_cols]
    df[symptom_cols] = df[symptom_cols].astype(np.int8)
    return df


# ══════════════════════════════════════════════════════════════════
# STEP 4 – VERIFICATION: print 5 sample rows for one disease
# ══════════════════════════════════════════════════════════════════

def print_verification_table(df: pd.DataFrame, disease: str = "Meningitis", n: int = 5):
    """
    Print n rows for a given disease showing only its active symptom columns,
    formatted as a compact Markdown table.
    """
    subset = df[df["label"] == disease].head(n)
    if subset.empty:
        print(f"[WARN] Disease '{disease}' not found in dataframe.")
        return

    # Only show columns that have at least one '1' in these 5 rows
    active_cols = [c for c in subset.columns
                   if c != "label" and subset[c].sum() > 0]
    # Cap for readability
    active_cols = active_cols[:20]

    display = subset[active_cols + ["label"]].reset_index(drop=True)

    # Build Markdown table manually
    header = "| # | " + " | ".join(c.replace("_", " ")[:25] for c in active_cols) + " | label |"
    sep    = "|---|" + "|".join(["---"] * len(active_cols)) + "|---|"
    print(f"\n### Verification: 5 sample rows for '{disease}'")
    print(header)
    print(sep)
    for i, (_, row) in enumerate(display.iterrows()):
        vals = " | ".join(str(int(row[c])) for c in active_cols)
        print(f"| {i+1} | {vals} | {row['label']} |")
    print()


# ══════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Medical RAG + XGBoost Pipeline")
    print("=" * 65)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/6] Loading source files…")
    templates, vocab, feature_dict, label_enc, class_dist = load_source_data()
    templates = filter_templates(templates)
    print(f"  Templates loaded : {len(templates)} diseases")
    print(f"  Symptom vocab    : {len(vocab)} terms")

    # ── Symptom columns ──────────────────────────────────────────
    print("\n[2/6] Resolving symptom columns from train.csv…")
    symptom_cols = get_symptom_columns_from_train()
    print(f"  Verified symptom columns : {len(symptom_cols)}")

    # ── Prevalence from train ─────────────────────────────────────
    print("\n[3/6] Computing data-driven symptom prevalence…")
    prevalence = compute_symptom_prevalence_from_train(symptom_cols)

    # ── Build RAG DB ─────────────────────────────────────────────
    print("\n[4/6] Building RAG database…")
    rag_db = build_rag_database(templates, vocab, prevalence, label_enc)
    with open(RAG_DB_OUT, "w", encoding="utf-8") as f:
        json.dump(rag_db, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {RAG_DB_OUT}")

    # ── RAG chunks ───────────────────────────────────────────────
    print("\n[5/6] Writing RAG Markdown chunks…")
    chunks = write_rag_chunks(rag_db, RAG_CHUNKS_DIR)

    # ── XGBoost dataset ──────────────────────────────────────────
    print("\n[6/6] Generating XGBoost training data…")
    disease_symptom_map = get_disease_symptom_map(rag_db, symptom_cols)
    df = generate_xgboost_dataset(rag_db, symptom_cols, disease_symptom_map)

    df.to_csv(XGBOOST_CSV_OUT, index=False)
    print(f"\n  ✓ XGBoost CSV saved → {XGBOOST_CSV_OUT}")
    print(f"    Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"    Diseases: {df['label'].nunique()}")
    print(f"    Samples per disease: {df.groupby('label').size().describe().to_dict()}")

    # ── Verification ─────────────────────────────────────────────
    print_verification_table(df, disease="Meningitis")
    print_verification_table(df, disease="Malaria")

    print("=" * 65)
    print("  Pipeline complete.")
    print(f"  → RAG DB       : {RAG_DB_OUT}")
    print(f"  → RAG Chunks   : {RAG_CHUNKS_DIR}/  ({len(chunks)} files)")
    print(f"  → XGBoost CSV  : {XGBOOST_CSV_OUT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
