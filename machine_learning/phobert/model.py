"""
PhoBERT Model definition for Sequence Classification.
Author: Hung (Deep Learning & PhoBERT)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .utils import ID2LABEL, LABEL2ID, get_device


class PhoBertSentimentClassifier(nn.Module):
    """
    Vietnamese Sentiment Classifier based on VinAI's PhoBERT.
    Wraps AutoModelForSequenceClassification with custom convenience methods.
    """

    DEFAULT_PRETRAINED: str = "vinai/phobert-base"

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_PRETRAINED,
        num_labels: int = 3,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """Standard forward pass delegating to Hugging Face model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate class probabilities via Softmax."""
        self.eval()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)

    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model weights and configuration in Hugging Face standard format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.config.save_pretrained(save_path)

    @classmethod
    def load_pretrained(
        cls,
        load_directory: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "PhoBertSentimentClassifier":
        """Load fine-tuned model checkpoint."""
        if device is None:
            device = get_device()
        classifier = cls(model_name_or_path=str(load_directory))
        classifier.to(device)
        classifier.eval()
        return classifier
