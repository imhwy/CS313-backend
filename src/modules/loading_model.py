"""
this class implements the interface for loading a pretrained model
"""

import os
from dotenv import load_dotenv

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")


class Model:
    """
    A wrapper class for loading and managing a pretrained model and tokenizer.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH
    ) -> None:
        """
        Initialize the PretrainedModel class with a specified model path.

        Args:
            model_path (str): Path to the pretrained model directory.
            Defaults to the MODEL_PATH environment variable.
        """
        self._model_path = model_path
        if not self._model_path:
            raise ValueError(
                "Model path must be provided either as an argument",
                "or through the MODEL_PATH environment variable."
            )

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model().to(self.device)

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer from the specified model path.

        Returns:
            PreTrainedTokenizer: The loaded tokenizer.
        """
        return AutoTokenizer.from_pretrained(self._model_path)

    def _load_model(self) -> PreTrainedModel:
        """
        Load the model for sequence classification from the specified model path.

        Returns:
            PreTrainedModel: The loaded model.
        """
        return AutoModelForSequenceClassification.from_pretrained(self._model_path)
