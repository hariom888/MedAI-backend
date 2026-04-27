"""
Medical Disease Prediction — XGBoost Training Pipeline
=======================================================
Trains on the combined dataset produced by:
  1. medical_pipeline.py     → output/train.csv, output/test.csv
  2. medical_rag_xgboost_pipeline.py → xgboost_training_data.csv  (RAG-enhanced rows)

Usage:
  python train.py                           # default: uses output/ CSVs
  python train.py --mode original           # only output/train.csv + output/test.csv
  python train.py --mode rag                # only xgboost_training_data.csv (80/20 split)
  python train.py --mode combined           # merge both datasets (default)
  python train.py --mode combined --out results/
  python train.py --no-shap                 # skip SHAP (faster on large datasets)
  python train.py --cv                      # run 5-fold cross-validation first

Outputs (written to --out directory, default: output/):
  medical_model.xgb           saved XGBoost model
  label_encoder.pkl           sklearn LabelEncoder (for inference)
  ordinal_encoder.pkl         sklearn OrdinalEncoder for cat cols
  classification_report.txt   per-class precision / recall / F1
  confusion_matrix.png        heatmap (top 25 diseases)
  feature_importance.png      top-30 gain importance bar chart
  shap_summary.png            SHAP mean |value| bar chart
  training_log.txt            timestamped run log
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # non-interactive backend — safe on headless servers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb

from excluded_diseases import is_excluded_disease

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train XGBoost medical classifier")
    p.add_argument("--mode", choices=["original", "rag", "combined"],
                   default="combined",
                   help="Which dataset(s) to use for training")
    p.add_argument("--original-train", default="output/train.csv",
                   help="Path to medical_pipeline.py train CSV")
    p.add_argument("--original-test",  default="output/test.csv",
                   help="Path to medical_pipeline.py test CSV")
    p.add_argument("--rag-csv",        default="xgboost_training_data.csv",
                   help="Path to RAG-pipeline XGBoost CSV")
    p.add_argument("--feature-dict",   default="output/feature_dictionary.json",
                   help="feature_dictionary.json from medical_pipeline.py")
    p.add_argument("--out",            default="output",
                   help="Output directory for model artifacts")
    p.add_argument("--n-estimators",   type=int, default=500)
    p.add_argument("--max-depth",      type=int, default=8)
    p.add_argument("--lr",             type=float, default=0.05,
                   help="XGBoost learning rate")
    p.add_argument("--subsample",      type=float, default=0.85)
    p.add_argument("--colsample",      type=float, default=0.80)
    p.add_argument("--cv",             action="store_true",
                   help="Run 5-fold stratified cross-validation before final fit")
    p.add_argument("--no-shap",        action="store_true",
                   help="Skip SHAP computation (saves time on very large datasets)")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--test-size",      type=float, default=0.20,
                   help="Test split fraction when RAG-only mode is used")
    return p.parse_args()


# ── Categorical columns present in the original pipeline output ───────────────

CAT_COLS = [
    "age_group", "sex", "bmi_group", "sleep_quality",
    "stress_level", "employment_status", "season", "region",
    "onset_speed", "row_type",
]

DROP_COLS = [
    "label", "row_type", "is_emergency", "psychosis_present",
    "self_harm_risk", "needs_specialist",
]


# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_original(train_path: str, test_path: str, logger: Logger):
    """Load the original medical_pipeline.py CSVs."""
    logger.log(f"Loading original train  : {train_path}")
    train = pd.read_csv(train_path, low_memory=False)
    logger.log(f"Loading original test   : {test_path}")
    test  = pd.read_csv(test_path,  low_memory=False)
    train = filter_excluded_labels(train, logger, "original train")
    test = filter_excluded_labels(test, logger, "original test")
    logger.log(f"  Original train shape  : {train.shape}")
    logger.log(f"  Original test shape   : {test.shape}")
    return train, test


def load_rag(rag_path: str, test_size: float, seed: int, logger: Logger):
    """Load RAG-pipeline CSV and do a stratified split."""
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
    """Ensure both DataFrames share the same column set (union, fill missing with 0)."""
    all_cols = sorted(set(df_a.columns) | set(df_b.columns))
    # Preserve label at end
    non_label = [c for c in all_cols if c != "label"]
    final_cols = non_label + ["label"]
    df_a = df_a.reindex(columns=final_cols, fill_value=0)
    df_b = df_b.reindex(columns=final_cols, fill_value=0)
    return df_a, df_b


def merge_datasets(
    orig_train, orig_test, rag_train, rag_test, logger: Logger
):
    """Merge original and RAG datasets, aligning columns."""
    logger.log("Aligning and merging datasets…")
    orig_train, rag_train = align_columns(orig_train, rag_train)
    orig_test,  rag_test  = align_columns(orig_test,  rag_test)

    train = pd.concat([orig_train, rag_train], ignore_index=True)
    test  = pd.concat([orig_test,  rag_test],  ignore_index=True)
    logger.log(f"  Combined train: {train.shape}  |  test: {test.shape}")
    return train, test


def filter_excluded_labels(df: pd.DataFrame, logger: Logger, dataset_name: str) -> pd.DataFrame:
    """Remove excluded diseases from a dataframe with a label column."""
    if "label" not in df.columns:
        return df

    mask = ~df["label"].astype(str).map(is_excluded_disease)
    removed = int((~mask).sum())
    if removed:
        logger.log(f"  Removed {removed} excluded rows from {dataset_name}.")
    return df.loc[mask].reset_index(drop=True)


# ── Pre-processing ────────────────────────────────────────────────────────────

def preprocess(train: pd.DataFrame, test: pd.DataFrame, logger: Logger, out_dir: Path):
    """
    Encode categoricals, label-encode target, split X/y.
    Returns X_train, X_test, y_train, y_test, label_encoder, feature_cols.
    """
    logger.log("Encoding categorical columns…")

    # Only encode cat cols that actually exist in this dataset
    present_cats = [c for c in CAT_COLS if c in train.columns]

    if present_cats:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train[present_cats] = oe.fit_transform(train[present_cats].astype(str))
        test[present_cats]  = oe.transform(test[present_cats].astype(str))
        with open(out_dir / "ordinal_encoder.pkl", "wb") as f:
            pickle.dump(oe, f)
        logger.log(f"  Encoded {len(present_cats)} categorical columns. Saved ordinal_encoder.pkl")
    else:
        logger.log("  No categorical columns found — skipping OrdinalEncoder.")

    # Label encode target
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"].astype(str))
    y_test  = le.transform(test["label"].astype(str))
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    logger.log(f"  Target classes        : {len(le.classes_)}")

    drop_present = [c for c in DROP_COLS if c in train.columns]
    feature_cols = [c for c in train.columns if c not in drop_present]

    X_train = train[feature_cols].fillna(0).astype(np.float32)
    X_test  = test[feature_cols].fillna(0).astype(np.float32)
    logger.log(f"  Feature columns       : {len(feature_cols)}")
    logger.log(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, le, feature_cols


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cv(X_train, y_train, model_params: dict, n_splits: int, logger: Logger):
    """Quick 5-fold stratified CV to sanity-check the model before final fit."""
    logger.log(f"Running {n_splits}-fold stratified cross-validation…")
    cv_model = xgb.XGBClassifier(**model_params, n_estimators=150)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=model_params["random_state"])
    scores = cross_val_score(cv_model, X_train, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1, verbose=0)
    logger.log(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    logger.log(f"  Per-fold   : {[f'{s:.4f}' for s in scores]}")
    return scores


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train, X_test, y_train, y_test,
    model_params: dict, n_estimators: int, logger: Logger, out_dir: Path
):
    """Fit XGBoost with early-stopping eval on test set."""
    logger.log(f"Training XGBoost  (n_estimators={n_estimators})…")

    model = xgb.XGBClassifier(
        **model_params,
        n_estimators=n_estimators,
        early_stopping_rounds=40,
        eval_metric="mlogloss",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    best_n = model.best_iteration + 1
    logger.log(f"  Best iteration        : {best_n}")

    model_path = out_dir / "medical_model.xgb"
    model.save_model(str(model_path))
    logger.log(f"  Model saved           : {model_path}")
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le: LabelEncoder, logger: Logger, out_dir: Path):
    """Classification report + confusion matrix."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)

    logger.log("\n=== Classification Report ===")
    for line in report.splitlines():
        logger.log(line)

    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(report)
    logger.log(f"\n  Report saved          : {report_path}")

    # Confusion matrix — top 25 classes by test frequency
    cm = confusion_matrix(y_test, y_pred)
    top_idx = pd.Series(y_test).value_counts().head(25).index.tolist()
    cm_sub  = cm[np.ix_(top_idx, top_idx)]
    class_names = le.classes_[top_idx]

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm_sub,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.3
    )
    ax.set_title("Confusion Matrix — Top 25 Diseases", fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.log(f"  Confusion matrix      : {cm_path}")

    return y_pred


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_cols: list, logger: Logger, out_dir: Path):
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


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(model, X_test, logger: Logger, out_dir: Path, max_samples: int = 500):
    try:
        import shap
    except ImportError:
        logger.log("  [WARN] shap not installed — skipping SHAP plot.")
        return

    logger.log("Computing SHAP values (this may take 1-2 min on large datasets)…")
    sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_vals, sample, show=False,
                      max_display=25, plot_type="bar")
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=12)
    plt.tight_layout()
    path = out_dir / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.log(f"  SHAP summary          : {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "training_log.txt")
    logger.log("=" * 60)
    logger.log("  Medical Disease Prediction — XGBoost Training")
    logger.log("=" * 60)
    logger.log(f"  Mode      : {args.mode}")
    logger.log(f"  Output dir: {out_dir}")
    logger.log(f"  Seed      : {args.seed}")

    # ── Load & prepare data ────────────────────────────────────────
    if args.mode == "original":
        train, test = load_original(args.original_train, args.original_test, logger)

    elif args.mode == "rag":
        train, test = load_rag(args.rag_csv, args.test_size, args.seed, logger)

    else:  # combined (default)
        orig_files_exist = (
            Path(args.original_train).exists() and
            Path(args.original_test).exists()
        )
        rag_exists = Path(args.rag_csv).exists()

        if orig_files_exist and rag_exists:
            orig_train, orig_test = load_original(args.original_train, args.original_test, logger)
            rag_train,  rag_test  = load_rag(args.rag_csv, args.test_size, args.seed, logger)
            train, test = merge_datasets(orig_train, orig_test, rag_train, rag_test, logger)
        elif orig_files_exist:
            logger.log("[WARN] RAG CSV not found — falling back to original only.")
            train, test = load_original(args.original_train, args.original_test, logger)
        elif rag_exists:
            logger.log("[WARN] Original CSVs not found — falling back to RAG only.")
            train, test = load_rag(args.rag_csv, args.test_size, args.seed, logger)
        else:
            logger.log("[ERROR] No data files found. Check --original-train / --rag-csv paths.")
            sys.exit(1)

    # ── Pre-processing ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test, le, feature_cols = preprocess(
        train, test, logger, out_dir
    )

    # ── Model hyperparameters ──────────────────────────────────────
    model_params = dict(
        max_depth        = args.max_depth,
        learning_rate    = args.lr,
        subsample        = args.subsample,
        colsample_bytree = args.colsample,
        n_jobs           = -1,
        random_state     = args.seed,
        tree_method      = "hist",          # fast on CPU; change to "gpu_hist" for GPU
        device           = "cpu",           # change to "cuda" for GPU
        num_class        = len(le.classes_),
        objective        = "multi:softprob",
    )
    logger.log(f"Model params: {model_params}")

    # ── Cross-validation (optional) ────────────────────────────────
    if args.cv:
        run_cv(X_train, y_train, model_params, n_splits=5, logger=logger)

    # ── Train ──────────────────────────────────────────────────────
    model = train_model(
        X_train, X_test, y_train, y_test,
        model_params, args.n_estimators, logger, out_dir
    )

    # ── Evaluate ───────────────────────────────────────────────────
    evaluate(model, X_test, y_test, le, logger, out_dir)
    plot_feature_importance(model, feature_cols, logger, out_dir)

    if not args.no_shap:
        plot_shap(model, X_test, logger, out_dir)
    else:
        logger.log("  SHAP skipped (--no-shap flag).")

    # ── Summary ────────────────────────────────────────────────────
    logger.log("\n" + "=" * 60)
    logger.log("  TRAINING COMPLETE")
    logger.log("=" * 60)
    logger.log(f"  Artifacts in: {out_dir}/")
    logger.log("    medical_model.xgb")
    logger.log("    label_encoder.pkl")
    logger.log("    ordinal_encoder.pkl  (if categorical cols present)")
    logger.log("    classification_report.txt")
    logger.log("    confusion_matrix.png")
    logger.log("    feature_importance.png")
    logger.log("    shap_summary.png     (unless --no-shap)")
    logger.log("    training_log.txt")
    logger.close()


if __name__ == "__main__":
    main()
