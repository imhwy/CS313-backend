"""
This service is responsible for prediction
"""

import torch

from src.modules.loading_model import Model
from src.modules.processing_data import TextProcessor


class CommentClassification:
    """
    A class for classifying comments into sentiment categories
    using a pretrained model and text processor.
    """

    def __init__(
        self,
        processor: TextProcessor = None,
        model: Model = None
    ) -> None:
        """
        Initialize the CommentClassification class.

        Args:
            processor (TextProcessor): An instance of TextProcessor for text preprocessing.
            model (PretrainedModel): An instance of PretrainedModel for text classification.
        """
        self._processor = processor
        self._model = model

    async def mapping(
        self,
        predict_label: int = 1
    ) -> None:
        """
        Map a predicted label to its corresponding sentiment category.

        Args:
            predict_label (int): The label predicted by the model.

        Returns:
            str: The sentiment category ("Negative", "Neutral", "Positive").
        """
        base_label = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }

        return base_label[predict_label]

    async def predict(
        self,
        text: str
    ) -> str:
        """
        Predict the sentiment of a given text.

        Args:
            text (str): The input text to classify.

        Returns:
            str: The sentiment category for the input text.
        """
        text = await self._processor.process_text(
            text=text
        )
        inputs = self._model.tokenizer(
            text=text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()

        result = await self.mapping(
            predict_label=prediction
        )

        return result
