"""
Utility functions for PhoBERT sentiment analysis.
Author: Hung (Deep Learning & PhoBERT)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS: List[str] = ["negative", "neutral", "positive"]
LABEL2ID: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

ALIASES: Dict[str, str] = {
    "negative": "negative",
    "neg": "negative",
    "0": "negative",
    "tiêu cực": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "1": "neutral",
    "trung lập": "neutral",
    "positive": "positive",
    "pos": "positive",
    "2": "positive",
    "tích cực": "positive",
}


def normalize_label(val: Union[str, int, float]) -> Optional[str]:
    """Map raw label representation to one of ['negative', 'neutral', 'positive']."""
    raw = str(val).strip().lower()
    if raw.endswith(".0") and raw[:-2] in ("0", "1", "2"):
        raw = raw[:-2]
    return ALIASES.get(raw)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Select appropriate computation device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(device)
        print(f"[Device] Using GPU: {device_name} ({device})")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available. Running on CPU.")
    return device


def compute_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
) -> Dict:
    """
    Compute comprehensive classification metrics matching baseline format:
    Accuracy, Macro/Weighted Precision, Recall, F1, Confusion Matrix, and
    detailed per-class reports.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    target_names = [ID2LABEL[i] for i in [0, 1, 2]]
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
    }


def save_confusion_matrix_plot(
    cm: Union[List[List[int]], np.ndarray],
    output_path: Union[str, Path],
    title: str = "PhoBERT Sentiment Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> None:
    """Save normalized confusion matrix heatmap to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if labels is None:
        labels = LABELS

    counts = np.asarray(cm, dtype=int)
    totals = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(
        counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals > 0
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = counts[i, j]
            pct = fraction[i, j]
            color = "white" if pct > 0.52 else "black"
            ax.text(
                j,
                i,
                f"{val:,}\n{pct:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold" if i == j else "normal",
            )

    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted sentiment",
        ylabel="Ground truth sentiment",
        title=title,
    )
    fig.colorbar(im, ax=ax, label="Fraction of true class", shrink=0.8)
    fig.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=200)
    plt.close(fig)
