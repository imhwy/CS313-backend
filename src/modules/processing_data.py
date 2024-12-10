"""
This class implements data processing for the given data source.
"""

import os
from typing import Dict
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

TEENCODE_PATH = os.getenv('TEENCODE_PATH')


class TextProcessor:
    """
    A class to process text with functionalities such as replacing teencode 
    and converting text to lowercase.
    """

    def __init__(
        self,
        teencode_path: str = TEENCODE_PATH
    ) -> None:
        """
        Initialize the TextProcessor with the path to the teencode mapping file.

        Args:
            teencode_path (str): Path to the teencode mapping file.
            Defaults to environment variable TEENCODE_PATH.
        """
        self._teencode_path = teencode_path
        self._config = {
            'replace_teencode': True,
            'to_lowercase': True
        }
        self._teencode_dict = self.load_teencode()

    def load_teencode(self) -> Dict[str, str]:
        """
        Load teencode mappings from a file into a dictionary.

        Returns:
            Dict[str, str]: A dictionary mapping teencode words to their replacements.
        """
        teencode_df = pd.read_csv(
            self._teencode_path,
            names=['teencode', 'map'],
            sep='\t'
        )
        return dict(zip(teencode_df['teencode'], teencode_df['map']))

    @staticmethod
    async def replace_teencode(
        text: str,
        teencode_dict: Dict[str, str]
    ) -> str:
        """
        Replace teencode words in a text using the provided mapping.

        Args:
            text (str): Input text to process.
            teencode_dict (Dict[str, str]): Dictionary of teencode mappings.

        Returns:
            str: Text with teencode words replaced.
        """
        return ' '.join(
            teencode_dict.get(
                word, word) for word in text.strip().split()
        )

    @staticmethod
    async def to_lowercase(text: str) -> str:
        """
        Convert the input text to lowercase.

        Args:
            text (str): Input text to process.

        Returns:
            str: Text converted to lowercase.
        """
        return text.lower()

    async def process_text(
        self,
        text: str
    ) -> str:
        """
        Process the input text based on the specified configuration.

        Args:
            text (str): Input text to process.
            teencode_dict (Dict[str, str], optional): Dictionary of teencode mappings.
            config (Dict[str, bool], optional): Configuration for processing steps.

        Returns:
            str: Processed text.
        """
        if self._config.get('replace_teencode') and self._teencode_dict:
            text = await self.replace_teencode(
                text=text,
                teencode_dict=self._teencode_dict
            )

        if self._config.get('to_lowercase'):
            text = await self.to_lowercase(text=text)

        return text
