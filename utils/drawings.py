import cv2
import numpy as np
from typing import List, Dict, Tuple


def draw_all_columns(img: np.ndarray, columns: List[Dict], box_width: int = 3) -> np.ndarray:
    """
    Draw all detected columns with different colored bounding boxes.

    Args:
        img: Input image
        columns: List of column dictionaries
        box_width: Width of bounding box lines

    Returns:
        Image with all columns drawn
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]

    result = img.copy()

    for idx, col in enumerate(columns):
        color = colors[idx % len(colors)]
        cv2.rectangle(result,
                     (col['x_start'], col['y_start']),
                     (col['x_end'], col['y_end']),
                     color, box_width)

        # Add column ID label
        label = f"Col {col['id']}"
        label_pos = (col['x_start'] + 5, col['y_start'] + 25)
        cv2.putText(result, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, color, 2)

    print(f"\n🎨 Drew {len(columns)} columns with colored bounding boxes")
    return result


def draw_clipped_ocr_boxes(img: np.ndarray, clipped_ocr: List[Tuple],
                           columns: List[Dict], box_width: int = 2) -> np.ndarray:
    """
    Draw all clipped OCR bounding boxes with column-matched colors.

    Args:
        img: Input image
        clipped_ocr: List of (bbox, text, confidence, column_id) tuples
        columns: List of column dictionaries
        box_width: Width of bounding box lines

    Returns:
        Image with OCR boxes drawn
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]

    result = img.copy()

    for (bbox, text, conf, col_id) in clipped_ocr:
        color = colors[col_id % len(colors)]
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

        # Optionally draw text label
        if text:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            text_pos = (int(min(x_coords)) + 2, int(min(y_coords)) + 12)
            cv2.putText(result, text[:20], text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

    print(f"🎨 Drew {len(clipped_ocr)} clipped OCR boxes")
    return result


def draw_outside_ocr_boxes(img: np.ndarray, outside_ocr: List[Tuple],
                          box_width: int = 3, color: Tuple = (0, 0, 255)) -> np.ndarray:
    """
    Draw OCR bounding boxes that are outside all columns.

    Args:
        img: Input image
        outside_ocr: List of (bbox, text, confidence) tuples for outside text
        box_width: Width of bounding box lines
        color: Color for outside boxes (default: red)

    Returns:
        Image with outside OCR boxes drawn
    """
    result = img.copy()

    for (bbox, text, conf) in outside_ocr:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

    print(f"🎨 Drew {len(outside_ocr)} outside OCR boxes in red")
    return result

