"""
Run a local disease prediction and print scaled confidence scores.

Usage:
  python test_prediction.py fever nausea vomiting
  python test_prediction.py "body pain" fever headache
"""

from __future__ import annotations

import argparse

from services import xgboost_service


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local disease model and print scaled confidence scores"
    )
    parser.add_argument(
        "symptoms",
        nargs="+",
        help="Symptoms to test, for example: fever nausea vomiting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = xgboost_service.predict_diseases(args.symptoms)

    print("Symptoms:")
    for symptom in args.symptoms:
        print(f"- {symptom}")

    print("\nPredictions:")
    if not predictions:
        print("No prediction passed the confidence threshold.")
        return

    for prediction in predictions:
        display_confidence = float(prediction.get("display_confidence", 0.0))
        print(
            f"{prediction['rank']}. "
            f"{prediction['disease_name']} -> "
            f"{display_confidence:.1f}"
        )


if __name__ == "__main__":
    main()
