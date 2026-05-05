"""
Medical Disease Prediction - XGBoost Training Pipeline.

Trains on the combined dataset produced by:
  1. medical_pipeline.py -> output/train.csv, output/test.csv
  2. medical_rag_xgboost_pipeline.py -> xgboost_training_data.csv

Key improvements:
  - Class-balanced sample weighting
  - Less aggressive XGBoost defaults to reduce overconfidence
  - Post-fit probability calibration on a held-out validation split
  - Engineered symptom features for overlap-heavy presentations
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from excluded_diseases import is_excluded_disease


CAT_COLS = [
    "age_group",
    "sex",
    "bmi_group",
    "sleep_quality",
    "stress_level",
    "employment_status",
    "season",
    "region",
    "onset_speed",
    "row_type",
]

DROP_COLS = [
    "label",
    "row_type",
    "is_emergency",
    "psychosis_present",
    "self_harm_risk",
    "needs_specialist",
]

MODEL_METADATA_NAME = "model_metadata.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the medical XGBoost classifier")
    parser.add_argument("--mode", choices=["original", "rag", "combined"], default="combined")
    parser.add_argument("--original-train", default="output/train.csv")
    parser.add_argument("--original-test", default="output/test.csv")
    parser.add_argument("--rag-csv", default="xgboost_training_data.csv")
    parser.add_argument("--out", default="output")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample", type=float, default=0.8)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    return parser.parse_args()


class Logger:
    def __init__(self, log_path: Path):
        self._log = open(log_path, "w", encoding="utf-8")
        self._start = time.time()

    def log(self, msg: str = ""):
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self._start
        line = f"[{ts} +{elapsed:6.1f}s] {msg}"
        print(line)
        self._log.write(line + "\n")
        self._log.flush()

    def close(self):
        self._log.close()


def filter_excluded_labels(df: pd.DataFrame, logger: Logger, dataset_name: str) -> pd.DataFrame:
    if "label" not in df.columns:
        return df

    mask = ~df["label"].astype(str).map(is_excluded_disease)
    removed = int((~mask).sum())
    if removed:
        logger.log(f"  Removed {removed} excluded rows from {dataset_name}.")
    return df.loc[mask].reset_index(drop=True)


def load_original(train_path: str, test_path: str, logger: Logger):
    logger.log(f"Loading original train  : {train_path}")
    train = pd.read_csv(train_path, low_memory=False)
    logger.log(f"Loading original test   : {test_path}")
    test = pd.read_csv(test_path, low_memory=False)
    train = filter_excluded_labels(train, logger, "original train")
    test = filter_excluded_labels(test, logger, "original test")
    logger.log(f"  Original train shape  : {train.shape}")
    logger.log(f"  Original test shape   : {test.shape}")
    return train, test


def load_rag(rag_path: str, test_size: float, seed: int, logger: Logger):
    logger.log(f"Loading RAG CSV         : {rag_path}")
    df = pd.read_csv(rag_path, low_memory=False)
    df = filter_excluded_labels(df, logger, "rag csv")
    logger.log(f"  RAG CSV shape         : {df.shape}")
    logger.log(f"  Diseases              : {df['label'].nunique()}")
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    logger.log(f"  RAG train: {train.shape}  |  RAG test: {test.shape}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def align_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = sorted(set(df_a.columns) | set(df_b.columns))
    non_label = [c for c in all_cols if c != "label"]
    final_cols = non_label + ["label"]
    return (
        df_a.reindex(columns=final_cols, fill_value=0),
        df_b.reindex(columns=final_cols, fill_value=0),
    )


def merge_datasets(orig_train, orig_test, rag_train, rag_test, logger: Logger):
    logger.log("Aligning and merging datasets...")
    orig_train, rag_train = align_columns(orig_train, rag_train)
    orig_test, rag_test = align_columns(orig_test, rag_test)
    train = pd.concat([orig_train, rag_train], ignore_index=True)
    test = pd.concat([orig_test, rag_test], ignore_index=True)
    logger.log(f"  Combined train: {train.shape}  |  test: {test.shape}")
    return train, test


def _is_binary_series(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    unique_values = set(values.astype(float).unique().tolist())
    return unique_values.issubset({0.0, 1.0})


def get_symptom_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    excluded = set(CAT_COLS) | set(DROP_COLS) | {"label", "symptom_count", "rarity_score"}
    columns = []
    for col in train.columns:
        if col in excluded or col not in test.columns:
            continue
        if _is_binary_series(train[col]) and _is_binary_series(test[col]):
            columns.append(col)
    return sorted(columns)


def select_interaction_pairs(
    train: pd.DataFrame,
    symptom_cols: list[str],
    max_base_symptoms: int = 40,
    max_pairs: int = 25,
    min_support: float = 0.01,
) -> list[list[str]]:
    if len(symptom_cols) < 2 or train.empty:
        return []

    prevalence = train[symptom_cols].mean().sort_values(ascending=False)
    base_cols = prevalence.head(min(max_base_symptoms, len(prevalence))).index.tolist()
    scored_pairs: list[tuple[float, str, str]] = []

    for idx, left in enumerate(base_cols):
        left_values = train[left].astype(np.float32)
        left_prev = float(left_values.mean())
        if left_prev <= 0:
            continue
        for right in base_cols[idx + 1 :]:
            right_values = train[right].astype(np.float32)
            right_prev = float(right_values.mean())
            if right_prev <= 0:
                continue

            joint = float((left_values * right_values).mean())
            if joint < min_support:
                continue

            lift = joint / max(left_prev * right_prev, 1e-8)
            scored_pairs.append((lift * joint, left, right))

    scored_pairs.sort(reverse=True)
    top_pairs = []
    for _, left, right in scored_pairs[:max_pairs]:
        top_pairs.append([left, right])
    return top_pairs


def engineer_features(
    train: pd.DataFrame, test: pd.DataFrame, logger: Logger
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = train.copy()
    test = test.copy()

    symptom_cols = get_symptom_columns(train, test)
    if not symptom_cols:
        logger.log("  [WARN] No binary symptom columns found for feature engineering.")
        return train, test, {"symptom_columns": [], "interaction_pairs": [], "rarity_weights": {}}

    symptom_prevalence = train[symptom_cols].mean().clip(0.0, 1.0)
    rarity_weights = symptom_prevalence.rsub(1.0).astype(np.float32)
    interaction_pairs = select_interaction_pairs(train, symptom_cols)

    train["symptom_count"] = train[symptom_cols].sum(axis=1).astype(np.float32)
    test["symptom_count"] = test[symptom_cols].sum(axis=1).astype(np.float32)

    train["rarity_score"] = (
        train[symptom_cols].astype(np.float32).mul(rarity_weights, axis=1).sum(axis=1).astype(np.float32)
    )
    test["rarity_score"] = (
        test[symptom_cols].astype(np.float32).mul(rarity_weights, axis=1).sum(axis=1).astype(np.float32)
    )

    for left, right in interaction_pairs:
        feature_name = f"pair__{left}__{right}"
        train[feature_name] = (
            train[left].astype(bool) & train[right].astype(bool)
        ).astype(np.float32)
        test[feature_name] = (
            test[left].astype(bool) & test[right].astype(bool)
        ).astype(np.float32)

    logger.log(f"  Symptom columns       : {len(symptom_cols)}")
    logger.log(f"  Interaction features  : {len(interaction_pairs)}")

    return train, test, {
        "symptom_columns": symptom_cols,
        "interaction_pairs": interaction_pairs,
        "rarity_weights": {name: float(weight) for name, weight in rarity_weights.items()},
    }


def preprocess(train: pd.DataFrame, test: pd.DataFrame, logger: Logger, out_dir: Path):
    logger.log("Engineering symptom features...")
    train, test, feature_metadata = engineer_features(train, test, logger)
    metadata_path = out_dir / MODEL_METADATA_NAME
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(feature_metadata, handle, indent=2)
    logger.log(f"  Feature metadata      : {metadata_path}")

    logger.log("Encoding categorical columns...")
    present_cats = [c for c in CAT_COLS if c in train.columns]
    if present_cats:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train[present_cats] = encoder.fit_transform(train[present_cats].astype(str))
        test[present_cats] = encoder.transform(test[present_cats].astype(str))
        with open(out_dir / "ordinal_encoder.pkl", "wb") as handle:
            pickle.dump(encoder, handle)
        logger.log(f"  Encoded {len(present_cats)} categorical columns. Saved ordinal_encoder.pkl")
    else:
        logger.log("  No categorical columns found - skipping OrdinalEncoder.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train["label"].astype(str))
    y_test = label_encoder.transform(test["label"].astype(str))
    with open(out_dir / "label_encoder.pkl", "wb") as handle:
        pickle.dump(label_encoder, handle)
    logger.log(f"  Target classes        : {len(label_encoder.classes_)}")

    drop_present = [c for c in DROP_COLS if c in train.columns]
    feature_cols = [c for c in train.columns if c not in drop_present]
    X_train = train[feature_cols].fillna(0).astype(np.float32)
    X_test = test[feature_cols].fillna(0).astype(np.float32)
    logger.log(f"  Feature columns       : {len(feature_cols)}")
    logger.log(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, label_encoder, feature_cols, feature_metadata


def run_cv(X_train, y_train, model_params: dict, n_splits: int, logger: Logger):
    logger.log(f"Running {n_splits}-fold stratified cross-validation...")
    cv_model = xgb.XGBClassifier(**model_params, n_estimators=150)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_params["random_state"])
    scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
    logger.log(f"  CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
    logger.log(f"  Per-fold   : {[f'{score:.4f}' for score in scores]}")
    return scores


def train_model(X_train, y_train, model_params: dict, n_estimators: int, logger: Logger, out_dir: Path):
    logger.log(f"Training XGBoost (n_estimators={n_estimators})...")

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    X_fit, X_cal, y_fit, y_cal, w_fit, _ = train_test_split(
        X_train,
        y_train,
        sample_weights,
        test_size=0.15,
        random_state=model_params["random_state"],
        stratify=y_train,
    )
    logger.log(f"  Base train split      : {X_fit.shape}")
    logger.log(f"  Calibration split     : {X_cal.shape}")

    model = xgb.XGBClassifier(
        **model_params,
        n_estimators=n_estimators,
        early_stopping_rounds=40,
        eval_metric="mlogloss",
    )
    model.fit(
        X_fit,
        y_fit,
        sample_weight=w_fit,
        eval_set=[(X_fit, y_fit), (X_cal, y_cal)],
        verbose=50,
    )

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        logger.log(f"  Best iteration        : {best_iteration + 1}")

    model_path = out_dir / "medical_model.xgb"
    model.save_model(str(model_path))
    logger.log(f"  Model saved           : {model_path}")

    calibration_method = "isotonic" if len(X_cal) >= 1000 else "sigmoid"
    logger.log(f"  Calibration method    : {calibration_method}")
    if FrozenEstimator is not None:
        calibrated_model = CalibratedClassifierCV(
            FrozenEstimator(model),
            method=calibration_method,
        )
    else:
        calibrated_model = CalibratedClassifierCV(
            model,
            method=calibration_method,
            cv="prefit",
        )
    calibrated_model.fit(X_cal, y_cal)

    calibrator_path = out_dir / "calibrated_model.pkl"
    with open(calibrator_path, "wb") as handle:
        pickle.dump(calibrated_model, handle)
    logger.log(f"  Calibrator saved      : {calibrator_path}")

    return model, calibrated_model


def evaluate(calibrated_model, X_test, y_test, label_encoder: LabelEncoder, logger: Logger, out_dir: Path):
    y_pred = calibrated_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=3)

    logger.log("\n=== Classification Report ===")
    for line in report.splitlines():
        logger.log(line)

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(f"Generated: {datetime.now()}\n\n")
        handle.write(report)
    logger.log(f"\n  Report saved          : {report_path}")

    matrix = confusion_matrix(y_test, y_pred)
    top_idx = pd.Series(y_test).value_counts().head(25).index.tolist()
    matrix_subset = matrix[np.ix_(top_idx, top_idx)]
    class_names = label_encoder.classes_[top_idx]

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        matrix_subset,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        linewidths=0.3,
    )
    ax.set_title("Confusion Matrix - Top 25 Diseases", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.log(f"  Confusion matrix      : {path}")


def plot_feature_importance(model, feature_cols: list[str], logger: Logger, out_dir: Path):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top30 = importances.nlargest(30).sort_values()

    fig, ax = plt.subplots(figsize=(11, 9))
    top30.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Top 30 Feature Importances (XGBoost gain)", fontsize=13)
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    path = out_dir / "feature_importance.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.log(f"  Feature importance    : {path}")


def plot_shap(model, X_test, logger: Logger, out_dir: Path, max_samples: int = 500):
    try:
        import shap
    except ImportError:
        logger.log("  [WARN] shap not installed - skipping SHAP plot.")
        return

    logger.log("Computing SHAP values (this may take 1-2 min on large datasets)...")
    sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=25, plot_type="bar")
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.log(f"  SHAP summary          : {path}")


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "training_log.txt")
    logger.log("=" * 60)
    logger.log("  Medical Disease Prediction - XGBoost Training")
    logger.log("=" * 60)
    logger.log(f"  Mode      : {args.mode}")
    logger.log(f"  Output dir: {out_dir}")
    logger.log(f"  Seed      : {args.seed}")

    if args.mode == "original":
        train, test = load_original(args.original_train, args.original_test, logger)
    elif args.mode == "rag":
        train, test = load_rag(args.rag_csv, args.test_size, args.seed, logger)
    else:
        original_exists = Path(args.original_train).exists() and Path(args.original_test).exists()
        rag_exists = Path(args.rag_csv).exists()
        if original_exists and rag_exists:
            orig_train, orig_test = load_original(args.original_train, args.original_test, logger)
            rag_train, rag_test = load_rag(args.rag_csv, args.test_size, args.seed, logger)
            train, test = merge_datasets(orig_train, orig_test, rag_train, rag_test, logger)
        elif original_exists:
            logger.log("[WARN] RAG CSV not found - falling back to original only.")
            train, test = load_original(args.original_train, args.original_test, logger)
        elif rag_exists:
            logger.log("[WARN] Original CSVs not found - falling back to RAG only.")
            train, test = load_rag(args.rag_csv, args.test_size, args.seed, logger)
        else:
            logger.log("[ERROR] No data files found. Check --original-train / --rag-csv paths.")
            sys.exit(1)

    X_train, X_test, y_train, y_test, label_encoder, feature_cols, _ = preprocess(
        train, test, logger, out_dir
    )

    model_params = dict(
        max_depth=min(args.max_depth, 4),
        learning_rate=min(args.lr, 0.05),
        subsample=min(args.subsample, 0.8),
        colsample_bytree=min(args.colsample, 0.8),
        reg_alpha=0.1,
        reg_lambda=1.5,
        min_child_weight=5,
        n_jobs=-1,
        random_state=args.seed,
        tree_method="hist",
        device="cpu",
        num_class=len(label_encoder.classes_),
        objective="multi:softprob",
        use_label_encoder=False,
    )
    logger.log(f"Model params: {model_params}")

    if args.cv:
        run_cv(X_train, y_train, model_params, n_splits=5, logger=logger)

    model, calibrated_model = train_model(
        X_train,
        y_train,
        model_params,
        args.n_estimators,
        logger,
        out_dir,
    )

    evaluate(calibrated_model, X_test, y_test, label_encoder, logger, out_dir)
    plot_feature_importance(model, feature_cols, logger, out_dir)

    if not args.no_shap:
        plot_shap(model, X_test, logger, out_dir)
    else:
        logger.log("  SHAP skipped (--no-shap flag).")

    logger.log("\n" + "=" * 60)
    logger.log("  TRAINING COMPLETE")
    logger.log("=" * 60)
    logger.log(f"  Artifacts in: {out_dir}/")
    logger.log("    medical_model.xgb")
    logger.log("    calibrated_model.pkl")
    logger.log("    label_encoder.pkl")
    logger.log("    ordinal_encoder.pkl  (if categorical cols present)")
    logger.log(f"    {MODEL_METADATA_NAME}")
    logger.log("    classification_report.txt")
    logger.log("    confusion_matrix.png")
    logger.log("    feature_importance.png")
    logger.log("    shap_summary.png     (unless --no-shap)")
    logger.log("    training_log.txt")
    logger.close()


if __name__ == "__main__":
    main()
