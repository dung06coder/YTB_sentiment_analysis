"""
Inference script for Vietnamese YouTube sentiment analysis using PhoBERT.
Author: Hung (Deep Learning & PhoBERT)

Usage:
    # 1. Predict a single comment from CLI
    python machine_learning/predict_phobert.py --text "Video này phân tích hay quá, ad làm thêm nhé 😍"

    # 2. Predict comments from a CSV file
    python machine_learning/predict_phobert.py --csv machine_learning/data/test.csv

    # 3. Python API for Backend integration (FastAPI / Flask)
    from machine_learning.predict_phobert import PhoBertSentimentPredictor
    predictor = PhoBertSentimentPredictor(model_dir="machine_learning/models/phobert_finetuned")
    res = predictor.predict_comment("Video đỉnh nóc kịch trần!")
    # res = {
    #     'label': 'positive',
    #     'confidence': 0.965,
    #     'probabilities': {'negative': 0.01, 'neutral': 0.025, 'positive': 0.965}
    # }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phobert.model import PhoBertSentimentClassifier
from phobert.utils import ID2LABEL, LABELS, get_device

try:
    from preprocessing import clean_text
except ImportError:
    from machine_learning.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PhoBERT-Predictor")


class PhoBertSentimentPredictor:
    """Production-ready inference wrapper for PhoBERT Vietnamese Sentiment Classifier."""

    DEFAULT_MODEL_DIR = ROOT / "models" / "phobert_finetuned"
    FALLBACK_PRETRAINED = "vinai/phobert-base"

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        max_length: int = 128,
        use_segment: bool = True,
    ):
        self.device = device or get_device()
        self.max_length = max_length
        self.use_segment = use_segment

        model_path = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR

        if model_path.exists() and (model_path / "config.json").exists():
            logger.info(f"Loading fine-tuned PhoBERT from local directory: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
            self.model = PhoBertSentimentClassifier.load_pretrained(
                str(model_path), device=self.device
            )
        else:
            logger.warning(
                f"Local checkpoint not found at '{model_path}'. "
                f"Falling back to pretrained base PhoBERT ({self.FALLBACK_PRETRAINED}) for testing."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.FALLBACK_PRETRAINED, use_fast=False
            )
            self.model = PhoBertSentimentClassifier(
                model_name_or_path=self.FALLBACK_PRETRAINED,
                num_labels=len(LABELS),
            ).to(self.device)
            self.model.eval()

    def preprocess(self, raw_text: str) -> str:
        """Clean raw comment using Vitou's pipeline and segment words."""
        return clean_text(raw_text, segment=self.use_segment)

    @torch.no_grad()
    def predict_comment(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict sentiment for a single raw Vietnamese comment.

        Returns
        -------
        dict with keys:
            - 'text': raw input text
            - 'clean_text': preprocessed & segmented text
            - 'label': predicted sentiment ('positive', 'neutral', 'negative')
            - 'confidence': confidence score (0.0 to 1.0)
            - 'probabilities': dict mapping each sentiment label to its probability
        """
        cleaned = self.preprocess(text)
        if not cleaned:
            # Empty text after cleaning defaults to neutral with low confidence
            return {
                "text": text,
                "clean_text": "",
                "label": "neutral",
                "confidence": 0.333,
                "probabilities": {"negative": 0.333, "neutral": 0.334, "positive": 0.333},
            }

        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        probs_tensor = self.model.predict_proba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).squeeze(0)

        probs = probs_tensor.cpu().numpy()
        pred_id = int(probs.argmax())
        pred_label = ID2LABEL[pred_id]
        confidence = float(probs[pred_id])

        prob_dict = {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}

        return {
            "text": text,
            "clean_text": cleaned,
            "label": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """Predict sentiment for a large list of comments (e.g. 500 crawled YouTube comments)."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            cleaned_chunk = [self.preprocess(t) for t in chunk]

            # Replace any empty strings with placeholder so tokenizer won't crash
            safe_chunk = [c if c else "bình_thường" for c in cleaned_chunk]

            encoded = self.tokenizer(
                safe_chunk,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)

            probs_tensor = self.model.predict_proba(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            probs = probs_tensor.cpu().numpy()

            for j, orig_text in enumerate(chunk):
                p = probs[j]
                pred_id = int(p.argmax())
                pred_label = ID2LABEL[pred_id]
                prob_dict = {
                    LABELS[k]: round(float(p[k]), 4) for k in range(len(LABELS))
                }
                results.append(
                    {
                        "text": orig_text,
                        "clean_text": cleaned_chunk[j],
                        "label": pred_label,
                        "confidence": round(float(p[pred_id]), 4),
                        "probabilities": prob_dict,
                    }
                )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict sentiment with fine-tuned PhoBERT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Raw Vietnamese YouTube comment")
    source.add_argument("--csv", type=Path, help="CSV file containing a 'text' column")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "phobert_finetuned",
        help="Path to saved PhoBERT directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path when using --csv",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for CSV inference",
    )

    args = parser.parse_args()
    predictor = PhoBertSentimentPredictor(model_dir=args.model_dir)

    if args.text is not None:
        result = predictor.predict_comment(args.text)
        print("\n" + "=" * 50)
        print("PHOBERT SENTIMENT INFERENCE RESULT:")
        print(f"  Comment     : {result['text']}")
        print(f"  Segmented   : {result['clean_text']}")
        print(f"  Sentiment   : {result['label'].upper()}")
        print(f"  Confidence  : {result['confidence']:.2%}")
        print("  Probabilities:")
        for label, prob in result["probabilities"].items():
            bar = "█" * int(prob * 20)
            print(f"    - {label.capitalize():<8}: {prob:.2%} | {bar}")
        print("=" * 50 + "\n")
        return

    # Process CSV
    csv_path = args.csv
    if not csv_path.is_file():
        logger.error(f"Input CSV not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Reading comments from {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "text" not in df.columns:
        logger.error("CSV must contain a 'text' column.")
        sys.exit(1)

    texts = df["text"].tolist()
    logger.info(f"Running inference on {len(texts):,} comments...")
    results = predictor.predict_batch(texts, batch_size=args.batch_size)

    df["predicted_label"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    df["prob_negative"] = [r["probabilities"]["negative"] for r in results]
    df["prob_neutral"] = [r["probabilities"]["neutral"] for r in results]
    df["prob_positive"] = [r["probabilities"]["positive"] for r in results]

    output_path = args.output or csv_path.with_name(csv_path.stem + "_phobert_predicted.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Done! Predictions saved to: {output_path}")

    # Summary statistics
    summary = df["predicted_label"].value_counts(normalize=True) * 100
    print("\nSummary of Predicted Sentiments:")
    for lab, pct in summary.items():
        print(f"  - {lab.capitalize()}: {pct:.2f}%")


if __name__ == "__main__":
    main()
