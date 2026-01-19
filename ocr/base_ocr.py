from abc import ABC, abstractmethod
from typing import Any, List, Tuple

class BaseOCR(ABC):
    """
    Abstract base class for OCR modules.
    """

    @abstractmethod
    def read_text(self, image: Any) -> str:
        """
        Extracts text from the given image.

        Args:
            image: The image object (format depends on implementation).

        Returns:
            str: Extracted text.
        """
        pass

    @abstractmethod
    def read_text_with_boxes(self, image: Any) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Extracts text and bounding boxes from the given image.

        Args:
            image: The image object.

        Returns:
            List of tuples containing (text, (x, y, w, h)).
        """
        pass

    @abstractmethod
    def detect_text_regions(self, image: Any) -> List[Tuple[int, int, int, int]]:
        """
        Detects text regions in the image.

        Args:
            image: The image object.

        Returns:
            List of bounding boxes (x, y, w, h).
        """
        pass