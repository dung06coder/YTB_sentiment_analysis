"""Predict Vietnamese YouTube sentiment with best_model.pkl.

From repository root:
    .\\.venv\\Scripts\\python.exe machine_learning\\predict_sentiment.py --text "video này hay quá 😍"
    .\\.venv\\Scripts\\python.exe machine_learning\\predict_sentiment.py --csv machine_learning\\data\\clean.csv

Requires preprocessing.py in machine_learning/ so the saved sklearn pipeline
can import the exact cleaning function used during training. Only load trusted
joblib/pickle files.
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Predict positive/neutral/negative sentiment")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Raw Vietnamese YouTube comment")
    source.add_argument("--csv", type=Path, help="CSV with a 'text' column")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "experiments/final_pseudo/models/best_model.pkl")
    parser.add_argument("--output", type=Path, help="CSV output location")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Saved model not found: {args.model}. Train it first or pass --model.")
    model = joblib.load(args.model)
    if args.text is not None:
        print("Comment:", args.text)
        print("Sentiment:", model.predict([args.text])[0])
        return
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "text" not in df.columns:
        parser.error("Input CSV must contain a 'text' column")
    df["predicted_label"] = model.predict(df["text"].tolist())
    target = args.output or args.csv.with_name(args.csv.stem + "_predicted.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} predictions -> {target}")


if __name__ == "__main__":
    main()
