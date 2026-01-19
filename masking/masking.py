from typing import List, Dict, Literal, Optional, Tuple
import cv2
import matplotlib.pyplot as plt

from utils.helper import mask_text_regions


BoundingBox = List[List[int]]  # [[x1, y1], [x2, y2]]
MaskedText = Tuple[str, BoundingBox]

def mask_texts_from_ocr_data(
    img_path: str,
    processed_data: List[Dict],
    mode: Literal["by_header", "by_text"],
    header_name: Optional[str] = None,
    target_text: Optional[str] = None,
):
    """
    Masks text regions in an image based on OCR-processed data.

    This function supports two masking strategies:
    1. 'by_header': Masks all texts under a specific header (excluding the header itself).
    2. 'by_text'  : Masks all occurrences of a specific target text across the image.

    Parameters
    ----------
    img_path : str
        Path to the input image where masking will be applied.

    processed_data : List[Dict]
        OCR output containing detected text blocks. Each item is expected to have:
        - 'header': str (only relevant for 'by_header' mode)
        - 'box': [x1, y1, x2, y2] (parent bounding box)
        - 'texts': list of dicts with:
            - 'text': str
            - 'box' : EasyOCR-style bounding box
              [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]

    mode : Literal["by_header", "by_text"]
        Determines how text is selected for masking.
        - 'by_header' → mask all texts under a given header
        - 'by_text'   → mask all matching target texts

    header_name : Optional[str], default=None
        Required when mode is 'by_header'.
        The header under which all texts will be masked.

    target_text : Optional[str], default=None
        Required when mode is 'by_text'.
        The exact text to be masked wherever it appears.

    Raises
    ------
    ValueError
        If required parameters for the selected mode are missing.
    """

    if mode == "by_header" and not header_name:
        raise ValueError("header_name must be provided when mode='by_header'")
    if mode == "by_text" and not target_text:
        raise ValueError("target_text must be provided when mode='by_text'")

    texts_to_mask = []

    for item in processed_data:
        parent_x1, parent_y1, _, _ = item["box"]
        header_text = item.get("header", "").strip()
        texts = item.get("texts", [])

        # Skip unrelated headers in by_header mode
        if mode == "by_header" and header_text != header_name.strip():
            continue

        for text_info in texts:
            if "text" not in text_info or "box" not in text_info:
                continue

            text_content = text_info["text"].strip()

            # Selection logic
            if mode == "by_header":
                if text_content == header_text:
                    continue
            elif mode == "by_text":
                if text_content != target_text.strip():
                    continue

            # Convert EasyOCR relative bbox to absolute coordinates
            points = text_info["box"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x1_abs = int(parent_x1 + min(x_coords))
            y1_abs = int(parent_y1 + min(y_coords))
            x2_abs = int(parent_x1 + max(x_coords))
            y2_abs = int(parent_y1 + max(y_coords))

            texts_to_mask.append(
                (text_content, [[x1_abs, y1_abs], [x2_abs, y2_abs]])
            )
    return mask_text_regions(img_path, texts_to_mask, True, True)