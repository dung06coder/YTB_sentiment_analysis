"""
Fine-tuning PhoBERT for Vietnamese YouTube Comment Sentiment Analysis.
Author: Hung (Deep Learning & PhoBERT)

Usage example (PowerShell):
    python machine_learning/train_phobert.py \\
        --train machine_learning/data/train.csv \\
        --test machine_learning/data/test.csv \\
        --epochs 4 --batch-size 16 --lr 2e-5

Google Colab:
    python machine_learning/train_phobert.py --fp16 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Ensure current directory and repository root are in Python path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phobert.dataset import SentimentDataset, create_data_loader
from phobert.model import PhoBertSentimentClassifier
from phobert.utils import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    compute_metrics,
    get_device,
    normalize_label,
    save_confusion_matrix_plot,
    set_seed,
)

try:
    from preprocessing import clean_text
except ImportError:
    # Fallback if imported from a different working directory
    from machine_learning.preprocessing import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PhoBERT-Trainer")


def load_and_preprocess_csv(
    file_path: Path,
    use_segment: bool = True,
    sample_size: int = None,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Read dataset, filter valid sentiment labels, and apply Vietnamese cleaning + segmentation.
    Uses Vitou's preprocessing.clean_text(segment=True).
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"File {file_path} must contain 'text' and 'label' columns.")

    # Filter and normalize labels
    df["label_norm"] = df["label"].map(normalize_label)
    invalid_count = df["label_norm"].isna().sum()
    if invalid_count > 0:
        logger.warning(f"Dropping {invalid_count} rows with invalid labels.")
        df = df.dropna(subset=["label_norm"])

    if sample_size and sample_size < len(df):
        logger.info(f"Subsampling {sample_size} rows for testing/quick run.")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info(f"Applying text preprocessing (segment={use_segment})...")
    # Clean text with Vitou's preprocessing pipeline
    processed_texts = [
        clean_text(t, segment=use_segment) for t in tqdm(df["text"], desc="Cleaning text")
    ]
    df["processed_text"] = processed_texts

    # Drop texts that ended up empty after cleaning
    valid_mask = df["processed_text"].str.strip() != ""
    df = df[valid_mask].reset_index(drop=True)

    labels = df["label_norm"].tolist()
    texts = df["processed_text"].tolist()

    logger.info(f"Loaded {len(texts):,} valid samples from {file_path.name}")
    for lab in LABELS:
        count = sum(1 for l in labels if l == lab)
        pct = (count / len(labels)) * 100 if len(labels) > 0 else 0
        logger.info(f"  Class '{lab}': {count:,} ({pct:.1f}%)")

    return texts, labels, df


def train_epoch(
    model: PhoBertSentimentClassifier,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    use_fp16: bool = False,
) -> float:
    """Run one training epoch with gradient updates."""
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and device.type == "cuda")

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(data_loader)


@torch.no_grad()
def evaluate_model(
    model: PhoBertSentimentClassifier,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict, List[int], List[int], List[List[float]]]:
    """Run evaluation and return metrics, ground truths, predictions, and prediction probabilities."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds)
    return metrics, all_targets, all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT for Vietnamese Sentiment Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data" / "train.csv",
        help="Path to training CSV file",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=ROOT / "data" / "test.csv",
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vinai/phobert-base",
        help="Pretrained PhoBERT identifier from Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "models" / "phobert_finetuned",
        help="Directory to save fine-tuned model and tokenizer",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for train and eval"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Peak learning rate for AdamW"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for regularization"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.1, help="Warmup portion of total training steps"
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Max sequence length for PhoBERT"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Enable automatic mixed precision on CUDA"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--quick-run",
        type=int,
        default=0,
        help="If > 0, subsample this many rows for fast testing pipeline",
    )
    parser.add_argument(
        "--no-segment",
        action="store_true",
        help="Skip word segmentation (not recommended for PhoBERT)",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    out_dir = args.output_dir
    rep_dir = args.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    logger.info(f"Loading PhoBERT tokenizer from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # 2. Data Loading & Preprocessing
    sample_sz = args.quick_run if args.quick_run > 0 else None
    use_segment = not args.no_segment

    train_texts, train_labels, train_df = load_and_preprocess_csv(
        args.train, use_segment=use_segment, sample_size=sample_sz
    )
    test_texts, test_labels, test_df = load_and_preprocess_csv(
        args.test, use_segment=use_segment, sample_size=sample_sz
    )

    # Split train into train & validation (90% train, 10% validation)
    from sklearn.model_selection import train_test_split

    indices = list(range(len(train_texts)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=args.seed,
        stratify=train_labels,
    )

    tr_texts = [train_texts[i] for i in train_idx]
    tr_labels = [train_labels[i] for i in train_idx]
    val_texts = [train_texts[i] for i in val_idx]
    val_labels = [train_labels[i] for i in val_idx]

    logger.info(
        f"Splits summary: Train={len(tr_texts):,}, Val={len(val_texts):,}, Test={len(test_texts):,}"
    )

    train_loader = create_data_loader(
        tr_texts,
        tr_labels,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
    )
    val_loader = create_data_loader(
        val_texts,
        val_labels,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )
    test_loader = create_data_loader(
        test_texts,
        test_labels,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False,
    )

    # 3. Model Initialization
    logger.info(f"Initializing PhoBERT Sequence Classifier from '{args.model_name}'...")
    model = PhoBertSentimentClassifier(
        model_name_or_path=args.model_name,
        num_labels=len(LABELS),
    ).to(device)

    # 4. Optimizer & Scheduler
    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # 5. Training Loop
    logger.info(f"Starting Fine-tuning: {args.epochs} epochs, {total_training_steps} steps.")
    best_val_macro_f1 = -1.0
    best_epoch = -1
    training_history = []

    start_train_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            use_fp16=args.fp16,
        )
        val_metrics, _, _, _ = evaluate_model(model, val_loader, device)
        epoch_duration = time.perf_counter() - epoch_start

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "duration_seconds": round(epoch_duration, 2),
            }
        )

        # Track and save best checkpoint based on validation Macro-F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            logger.info(
                f"--> New best model found at epoch {epoch} (Val Macro-F1: {best_val_macro_f1:.4f}). Saving..."
            )
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    total_training_time = round(time.perf_counter() - start_train_time, 2)
    logger.info(
        f"Training completed in {total_training_time}s. Best epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}"
    )

    # 6. Evaluation on Held-out Test Set
    logger.info("Loading best fine-tuned model for final Test set evaluation...")
    best_model = PhoBertSentimentClassifier.load_pretrained(out_dir, device=device)
    test_metrics, true_ids, pred_ids, pred_probs = evaluate_model(
        best_model, test_loader, device
    )

    logger.info("=" * 60)
    logger.info("PHOBERT TEST SET EVALUATION RESULTS:")
    logger.info(f"  Test Accuracy   : {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Macro-F1   : {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Test Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    logger.info("=" * 60)

    # Print detailed classification report
    print("\nDetailed Test Classification Report:")
    report_dict = test_metrics["classification_report"]
    report_df = pd.DataFrame(report_dict).transpose()
    print(report_df.round(4).to_string())

    # 7. Save Metrics, Reports & Plots
    metrics_path = rep_dir / "phobert_metrics.json"
    full_report = {
        "model": "PhoBERT (vinai/phobert-base) Fine-tuned",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_params": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "seed": args.seed,
            "total_training_time_seconds": total_training_time,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
        },
        "evaluation": test_metrics,
        "classes": LABELS,
        "label2id": LABEL2ID,
        "history": training_history,
        "device": str(device),
    }

    metrics_path.write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved PhoBERT test metrics to {metrics_path}")

    # Save Confusion Matrix CSV & Heatmap Plot
    cm_csv_path = rep_dir / "phobert_confusion_matrix.csv"
    pd.DataFrame(test_metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
        cm_csv_path, encoding="utf-8-sig"
    )

    cm_png_path = rep_dir / "phobert_confusion_matrix.png"
    save_confusion_matrix_plot(
        test_metrics["confusion_matrix"],
        cm_png_path,
        title="PhoBERT Sentiment Confusion Matrix",
    )
    logger.info(f"Saved confusion matrix plot to {cm_png_path}")

    # Save Error Analysis CSV (Misclassified comments)
    pred_labels = [ID2LABEL[p] for p in pred_ids]
    test_df_copy = test_df.copy()
    test_df_copy["predicted_label"] = pred_labels
    test_df_copy["pred_probability_negative"] = [round(p[0], 4) for p in pred_probs]
    test_df_copy["pred_probability_neutral"] = [round(p[1], 4) for p in pred_probs]
    test_df_copy["pred_probability_positive"] = [round(p[2], 4) for p in pred_probs]

    errors = test_df_copy[test_df_copy["label_norm"] != test_df_copy["predicted_label"]]
    errors_path = rep_dir / "phobert_errors.csv"
    errors.to_csv(errors_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(errors):,} misclassified error cases to {errors_path}")

    # 8. Remind user about large weight files & Google Drive
    logger.info("=" * 70)
    logger.info("QUAN TRỌNG (LƯU Ý CỦA NHÓM TRƯỞNG & ĐỒ ÁN):")
    logger.info(f"- Thư mục model đã lưu: {out_dir}")
    logger.info("- File trọng số (.pt / .bin / .safetensors) rất nặng (~540MB).")
    logger.info("- KHÔNG đẩy các file trọng số lên GitHub (đã cấu hình .gitignore).")
    logger.info("- Hãy nén thư mục model và tải lên Google Drive chung của nhóm.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
