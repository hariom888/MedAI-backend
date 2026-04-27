"""Filter current root-level dataset artifacts using the shared exclusion list."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from excluded_diseases import EXCLUDED_DISEASES, is_excluded_disease


ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def filter_json_assets() -> None:
    templates_path = ROOT / "disease_templates.json"
    if templates_path.exists():
        templates = json.loads(templates_path.read_text(encoding="utf-8"))
        templates = {
            name: value for name, value in templates.items()
            if not is_excluded_disease(name)
        }
        _write_json(templates_path, templates)

    class_dist_path = ROOT / "class_distribution.json"
    if class_dist_path.exists():
        class_dist = json.loads(class_dist_path.read_text(encoding="utf-8"))
        class_dist = {
            name: value for name, value in class_dist.items()
            if not is_excluded_disease(name)
        }
        _write_json(class_dist_path, class_dist)

    rag_db_path = ROOT / "rag_disease_db.json"
    if rag_db_path.exists():
        rag_db = json.loads(rag_db_path.read_text(encoding="utf-8"))
        rag_db = [
            row for row in rag_db
            if not is_excluded_disease(row.get("disease_name") or row.get("disease", ""))
        ]
        _write_json(rag_db_path, rag_db)

    label_encoder_path = ROOT / "label_encoder.json"
    if label_encoder_path.exists():
        enc = json.loads(label_encoder_path.read_text(encoding="utf-8"))
        classes = [name for name in enc.get("classes", []) if not is_excluded_disease(name)]
        remapped = {name: idx for idx, name in enumerate(classes)}
        enc["classes"] = classes
        if "label_to_int" in enc:
            enc["label_to_int"] = remapped
        if "label_to_id" in enc:
            enc["label_to_id"] = remapped
        if "int_to_label" in enc:
            enc["int_to_label"] = {str(idx): name for idx, name in enumerate(classes)}
        if "id_to_label" in enc:
            enc["id_to_label"] = {str(idx): name for idx, name in enumerate(classes)}
        _write_json(label_encoder_path, enc)

    feature_dict_path = ROOT / "feature_dictionary.json"
    if feature_dict_path.exists():
        feature_dict = json.loads(feature_dict_path.read_text(encoding="utf-8"))
        filtered = {}
        for symptom, diseases in feature_dict.items():
            kept = [name for name in diseases if not is_excluded_disease(name)]
            if kept:
                filtered[symptom] = kept
        _write_json(feature_dict_path, filtered)


def filter_csv_assets() -> None:
    for filename, disease_col in [
        ("train.csv", "disease"),
        ("test.csv", "disease"),
        ("xgboost_training_data.csv", "label"),
    ]:
        path = ROOT / filename
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)
        if disease_col in df.columns:
            df = df[~df[disease_col].astype(str).map(is_excluded_disease)].reset_index(drop=True)
        if disease_col == "disease" and "label" in df.columns and "disease" in df.columns:
            classes = sorted(df["disease"].astype(str).unique())
            remapped = {name: idx for idx, name in enumerate(classes)}
            df["label"] = df["disease"].astype(str).map(remapped)
        df.to_csv(path, index=False)


def filter_rag_chunks() -> None:
    rag_dir = ROOT / "rag_chunks"
    if not rag_dir.exists():
        return

    excluded_stems = {
        "abdominal_aortic_aneurysm",
        "acute_respiratory_infection",
        "adenomyosis",
        "allergic_rhinitis",
        "anal_cancer",
        "ankle_sprain",
        "arterial_thrombosis",
        "bronchitis",
        "cardiomyopathy",
        "central_nervous_system_tumor",
        "chronic_kidney_disease",
        "chronic_obstructive_pulmonary_disease",
        "cirrhosis",
        "colorectal_cancer",
        "coronary_artery_disease",
        "deep_vein_thrombosis",
        "endocarditis",
        "fatty_liver_disease",
        "h1n1_swine_flu",
        "heart_failure",
        "lung_cancer",
        "melasma",
        "middle_east_respiratory_syndrome",
        "myocarditis",
        "occupational_lung_disease",
        "osteomyelitis",
        "osteoporosis",
        "pericarditis",
        "peripheral_artery_disease",
        "pleural_effusion",
        "polycystic_ovary_syndrome",
        "rheumatic_heart_disease",
        "scoliosis",
        "severe_acute_respiratory_syndrome",
        "skin_cancer",
    }
    for path in rag_dir.glob("*.md"):
        stem = path.stem.lower()
        stem = stem.replace("_chunk_1", "").replace("_chunk_2", "")
        if stem in excluded_stems:
            path.unlink(missing_ok=True)


def main() -> None:
    filter_json_assets()
    filter_csv_assets()
    filter_rag_chunks()
    print(f"Applied exclusions for {len(EXCLUDED_DISEASES)} diseases.")


if __name__ == "__main__":
    main()
