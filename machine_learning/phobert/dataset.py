"""
PyTorch Dataset and DataLoader preparation for PhoBERT Sentiment Classification.
Author: Hung (Deep Learning & PhoBERT)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .utils import LABEL2ID, normalize_label


class SentimentDataset(Dataset):
    """
    PyTorch Dataset wrapper for Vietnamese YouTube comments.
    Encodes text into token IDs and attention masks using PhoBERT Tokenizer.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[Union[str, int]]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
    ):
        """
        Parameters
        ----------
        texts : List[str]
            List of processed/segmented Vietnamese comment strings.
        labels : Optional[List[Union[str, int]]]
            Sentiment labels. Can be raw strings ('negative', 'neutral', 'positive')
            or integer IDs (0, 1, 2). Omit for inference.
        tokenizer : AutoTokenizer
            Hugging Face PhoBERT tokenizer instance.
        max_length : int
            Maximum token sequence length (PhoBERT supports up to 256).
        """
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.max_length = max_length
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            # Fallback to default pretrained PhoBERT tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vinai/phobert-base", use_fast=False
            )

        self.labels = None
        if labels is not None:
            encoded_labels = []
            for lab in labels:
                if isinstance(lab, (int, torch.Tensor)):
                    encoded_labels.append(int(lab))
                else:
                    norm = normalize_label(lab)
                    if norm is None or norm not in LABEL2ID:
                        raise ValueError(f"Invalid sentiment label encountered: {lab}")
                    encoded_labels.append(LABEL2ID[norm])
            self.labels = torch.tensor(encoded_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item


def create_data_loader(
    texts: List[str],
    labels: Optional[List[Union[str, int]]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    batch_size: int = 16,
    max_length: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience helper to build a PyTorch DataLoader."""
    dataset = SentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
