"""
Evaluate probability calibration for the trained medical classifier.

Usage:
  python check_calibration.py
  python check_calibration.py --artifacts output --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

from train import Logger, load_original, load_rag, merge_datasets, preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Check calibration of the trained classifier")
    parser.add_argument("--artifacts", default="output", help="Directory containing trained artifacts")
    parser.add_argument("--original-train", default="output/train.csv", help="Path to original train CSV")
    parser.add_argument("--original-test", default="output/test.csv", help="Path to original test CSV")
    parser.add_argument("--rag-csv", default="xgboost_training_data.csv", help="Path to RAG CSV")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split for RAG-only mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_eval_data(args, logger: Logger):
    original_exists = Path(args.original_train).exists() and Path(args.original_test).exists()
    rag_exists = Path(args.rag_csv).exists()

    if original_exists and rag_exists:
        orig_train, orig_test = load_original(args.original_train, args.original_test, logger)
        rag_train, rag_test = load_rag(args.rag_csv, args.test_size, args.seed, logger)
        return merge_datasets(orig_train, orig_test, rag_train, rag_test, logger)
    if original_exists:
        return load_original(args.original_train, args.original_test, logger)
    if rag_exists:
        return load_rag(args.rag_csv, args.test_size, args.seed, logger)
    raise FileNotFoundError("No evaluation dataset found. Expected original CSVs or xgboost_training_data.csv.")


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts)
    calibrator_path = artifacts_dir / "calibrated_model.pkl"

    if not calibrator_path.exists():
        raise FileNotFoundError(
            f"Missing calibrated model at {calibrator_path}. Run training first: "
            f"python train.py --mode combined --out {artifacts_dir}"
        )

    logger = Logger(Path("calibration_check.log"))
    train_df, test_df = load_eval_data(args, logger)

    _, X_test, _, y_test, _, _, _ = preprocess(train_df, test_df, logger, artifacts_dir)

    with open(calibrator_path, "rb") as handle:
        model = pickle.load(handle)

    probabilities = model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    ll = log_loss(y_test, probabilities)
    y_onehot = np.zeros_like(probabilities)
    y_onehot[np.arange(len(y_test)), y_test] = 1
    brier = np.mean(np.sum((probabilities - y_onehot) ** 2, axis=1))
    top1_accuracy = float((predictions == y_test).mean())

    print(f"log_loss={ll:.4f}")
    print(f"multiclass_brier={brier:.4f}")
    print(f"top1_accuracy={top1_accuracy:.4f}")

    bins = np.linspace(0.0, 1.0, 6)
    print("\nconfidence_bins:")
    for idx in range(len(bins) - 1):
        lo, hi = bins[idx], bins[idx + 1]
        if idx < len(bins) - 2:
            mask = (confidences >= lo) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences <= hi)

        if mask.sum() == 0:
            continue

        accuracy = float((predictions[mask] == y_test[mask]).mean())
        avg_confidence = float(confidences[mask].mean())
        print(
            f"bin {lo:.1f}-{hi:.1f}: "
            f"n={int(mask.sum())} avg_conf={avg_confidence:.3f} acc={accuracy:.3f}"
        )

    logger.close()


if __name__ == "__main__":
    main()
