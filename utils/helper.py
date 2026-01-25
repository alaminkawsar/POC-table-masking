from typing import List, Tuple, Literal
import cv2
import matplotlib.pyplot as plt
import numpy as np

BoundingBox = List[List[int]]  # [[x1, y1], [x2, y2]]
MaskedText = Tuple[str, BoundingBox]

def _generate_masked_text(text: str) -> str:
    """Generate a masked version of the text using asterisks."""
    return "*" * len(text)


def _normalize_bbox(bbox: BoundingBox) -> Tuple[int, int, int, int]:
    """Convert bbox points to normalized (x1, y1, x2, y2)."""
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))


def _is_valid_bbox(x1: int, y1: int, x2: int, y2: int) -> bool:
    """Check if bounding box coordinates are valid."""
    return x2 > x1 and y2 > y1


def _draw_centered_text(
    image,
    text: str,
    bbox: Tuple[int, int, int, int],
    font,
    font_scale: float,
    color: tuple,
    thickness: int,
) -> None:
    """Draw text centered inside a bounding box."""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    text_x = x1 + max(0, (bbox_width - text_width) // 2)
    text_y = y1 + max(text_height, (bbox_height + text_height) // 2)

    text_y = min(text_y, y2 - baseline)

    cv2.putText(
        image,
        text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def mask_text_regions(
    image_path: str,
    texts: List[MaskedText],
    mask_mode: Literal["white", "blur", "pixelate"] = "white",
    draw_bbox: bool = False,
    blur_kernel: Tuple[int, int] = (21, 21),
    pixelate_scale: float = 0.1,
) -> np.ndarray:
    """
    Mask text regions in an image using OpenCV only.

    Parameters
    ----------
    image_path : str
        Path to input image.

    texts : List[Tuple[str, BoundingBox]]
        List of detected texts and bounding boxes as:
        ("text", (x1, y1, x2, y2))

    mask_mode : {"white", "blur", "pixelate"}, default="white"
        Masking strategy to apply.

    draw_bbox : bool, default=False
        Draw bounding box outlines.

    blur_kernel : Tuple[int, int], default=(21, 21)
        Kernel size used when mask_mode="blur".

    pixelate_scale : float, default=0.1
        Downscale factor for pixelation (0 < scale < 1).

    Returns
    -------
    np.ndarray
        Masked image (BGR).
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    height, width = image.shape[:2]

    for text_content, points in texts:
        masked_text = "*" * (len(text_content) // 2) + "*" * (len(text_content) % 2)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Calculate absolute coordinates by adding parent box offset
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = image[y1:y2, x1:x2]

        if mask_mode == "white":
            roi[:] = (255, 255, 255)

        elif mask_mode == "blur":
            roi[:] = cv2.GaussianBlur(roi, blur_kernel, 0)

        elif mask_mode == "pixelate":
            roi[:] = _pixelate_region(roi, pixelate_scale)

        if draw_bbox:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return image


def _pixelate_region(region, scale: float):
    """Pixelate an image region."""
    h, w = region.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    temp = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def _display_image(image) -> None:
    """Display image using matplotlib."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Masked Text Regions")
    plt.show()
    
    
import numpy as np

def easyocr_bbox_to_xyxy(easyocr_bbox):
    """Converts an EasyOCR bounding box format to (x_min, y_min, x_max, y_max)."""
    # EasyOCR bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    x_coords = [p[0] for p in easyocr_bbox]
    y_coords = [p[1] for p in easyocr_bbox]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    return (x_min, y_min, x_max, y_max)

def compute_iou(boxA, boxB):
    """Calculates the Intersection Over Union (IOU) of two bounding boxes.
    Boxes are expected in (x_min, y_min, x_max, y_max) format.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    inter_width = xB - xA
    inter_height = yB - yA
    if inter_width < 0 or inter_height < 0:
        return 0.0

    interArea = inter_width * inter_height

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the IOU value
    return iou

