"""
This service is responsible for managing the configuration
"""

from src.modules.loading_model import Model
from src.modules.processing_data import TextProcessor
from src.services.classify_comment import CommentClassification


class Service:
    """
    A service layer to manage dependencies for comment classification,
    including model loading and text processing.
    """

    def __init__(self):
        """
        Initialize the Service class and its components.

        Components initialized:
            - PreTrainedModel: Loads the pretrained model.
            - TextProcessor: Handles preprocessing.
            - CommentClassification: Combines model and text processor.
        """
        self._loading_model = Model()
        self._text_processor = TextProcessor()
        self._comment_classification = CommentClassification(
            processor=self._text_processor,
            model=self._loading_model
        )

    @property
    def comment_classification(self) -> CommentClassification:
        """
        Get the CommentClassification instance.

        Returns:
            CommentClassification: The instance used 
            for comment classification tasks.
        """
        return self._comment_classification

    @property
    def loading_model(self) -> Model:
        """
        Get the PreTrainedModel instance.

        Returns:
            PreTrainedModel: The instance responsible for 
            loading and managing the pretrained model.
        """
        return self._loading_model

    @property
    def text_processor(self) -> TextProcessor:
        """
        Get the TextProcessor instance.

        Returns:
            TextProcessor: The instance used for text preprocessing.
        """
        return self._text_processor
