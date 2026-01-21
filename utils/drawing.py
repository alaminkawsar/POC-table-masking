import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple



BoundingBox = List[List[int]]  # [[x1, y1], [x2, y2]]
MaskedText = Tuple[str, BoundingBox]

def draw_box_on_all_texts(image_path: str, all_processed_data: list, draw_bbox: bool = True, fill_bbox_white: bool = False):
    # Load the original image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Define visual properties
    box_color = (0, 255, 0)  # Green in BGR
    text_color = (0, 0, 255) # Red in BGR
    white_color = (255, 255, 255) # White in BGR for filling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    for processed_data_item in all_processed_data:
        parent_x1, parent_y1, _, _ = processed_data_item["box"]
        texts_to_annotate = processed_data_item["texts"]

        for text_info in texts_to_annotate:
            if "text" in text_info and "box" in text_info:
                text_content = text_info["text"]
                # Create masked text with asterisks
                masked_text = "*" * (len(text_content) // 2) + "*" * (len(text_content) % 2)

                # EasyOCR bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # Extract the corners of the EasyOCR bounding box (relative to parent box)
                points = text_info["box"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                # Calculate absolute coordinates by adding parent box offset
                x1_abs = int(parent_x1 + min(x_coords))
                y1_abs = int(parent_y1 + min(y_coords))
                x2_abs = int(parent_x1 + max(x_coords))
                y2_abs = int(parent_y1 + max(y_coords))

                # Ensure valid bounding box (min_x < max_x, min_y < max_y) before proceeding
                if x2_abs <= x1_abs or y2_abs <= y1_abs:
                    print(f"Warning: Invalid bounding box for text '{text_content}'. Skipping.")
                    continue

                # Fill the bounding box with white if the flag is true
                if fill_bbox_white:
                    cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), white_color, cv2.FILLED)

                # Draw the bounding box outline if the flag is true
                if draw_bbox:
                    cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), box_color, thickness)

                # Get text size to center it
                (text_width, text_height), baseline = cv2.getTextSize(masked_text, font, font_scale, thickness)

                bbox_width = x2_abs - x1_abs
                bbox_height = y2_abs - y1_abs

                # Calculate centered horizontal position
                ideal_text_x = x1_abs + (bbox_width - text_width) // 2

                # Clamp text_x to ensure it stays within the horizontal bounds of the bbox
                text_x = max(x1_abs, min(ideal_text_x, x2_abs - text_width))

                # Calculate centered vertical position (baseline)
                # The overall vertical extent of the text (from top of ascenders to bottom of descenders) is text_height + baseline.
                # We want the center of this overall text extent to align with the center of the bounding box.
                bbox_center_y = y1_abs + bbox_height // 2

                # Calculate the ideal baseline y-coordinate such that the *vertical center* of the entire text string
                # (including ascenders and descenders) aligns with the center of the bounding box.
                ideal_text_y_baseline = bbox_center_y + (text_height - baseline) // 2

                # Clamp text_y to ensure it stays within vertical bounds
                # The text's top edge is at (text_y - text_height). Must be >= y1_abs. So text_y >= y1_abs + text_height.
                # The text's bottom edge is at (text_y + baseline). Must be <= y2_abs. So text_y <= y2_abs - baseline.
                text_y = max(y1_abs + text_height, min(ideal_text_y_baseline, y2_abs - baseline))

                # Put the masked text on the image
                # cv2.putText(img, masked_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Convert BGR image to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Image with All Extracted Texts Annotated and Masked')
    plt.savefig('annotated_all_texts.png', dpi=300, bbox_inches='tight')
    # Save the figure to a file
    # Display the plot (optional, but must come after savefig to avoid blank images)
    plt.show()
    
def mask_all_extracted_texts(image_path: str, all_processed_data: list, draw_bbox: bool = True, fill_bbox_white: bool = False):
    # Load the original image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Define visual properties
    box_color = (0, 255, 0)  # Green in BGR
    text_color = (0, 0, 255) # Red in BGR
    white_color = (255, 255, 255) # White in BGR for filling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    for processed_data_item in all_processed_data:
        parent_x1, parent_y1, _, _ = processed_data_item["box"]
        texts_to_annotate = processed_data_item["texts"]

        for text_info in texts_to_annotate:
            if "text" in text_info and "box" in text_info:
                text_content = text_info["text"]
                # Create masked text with asterisks
                masked_text = "*" * (len(text_content) // 2) + "*" * (len(text_content) % 2)

                # EasyOCR bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # Extract the corners of the EasyOCR bounding box (relative to parent box)
                points = text_info["box"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                # Calculate absolute coordinates by adding parent box offset
                x1_abs = int(parent_x1 + min(x_coords))
                y1_abs = int(parent_y1 + min(y_coords))
                x2_abs = int(parent_x1 + max(x_coords))
                y2_abs = int(parent_y1 + max(y_coords))

                # Ensure valid bounding box (min_x < max_x, min_y < max_y) before proceeding
                if x2_abs <= x1_abs or y2_abs <= y1_abs:
                    print(f"Warning: Invalid bounding box for text '{text_content}'. Skipping.")
                    continue

                # Fill the bounding box with white if the flag is true
                if fill_bbox_white:
                    cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), white_color, cv2.FILLED)

                # Draw the bounding box outline if the flag is true
                if draw_bbox:
                    cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), box_color, thickness)

                # Get text size to center it
                (text_width, text_height), baseline = cv2.getTextSize(masked_text, font, font_scale, thickness)

                bbox_width = x2_abs - x1_abs
                bbox_height = y2_abs - y1_abs

                # Calculate centered horizontal position
                ideal_text_x = x1_abs + (bbox_width - text_width) // 2

                # Clamp text_x to ensure it stays within the horizontal bounds of the bbox
                text_x = max(x1_abs, min(ideal_text_x, x2_abs - text_width))

                # Calculate centered vertical position (baseline)
                # The overall vertical extent of the text (from top of ascenders to bottom of descenders) is text_height + baseline.
                # We want the center of this overall text extent to align with the center of the bounding box.
                bbox_center_y = y1_abs + bbox_height // 2

                # Calculate the ideal baseline y-coordinate such that the *vertical center* of the entire text string
                # (including ascenders and descenders) aligns with the center of the bounding box.
                ideal_text_y_baseline = bbox_center_y + (text_height - baseline) // 2

                # Clamp text_y to ensure it stays within vertical bounds
                # The text's top edge is at (text_y - text_height). Must be >= y1_abs. So text_y >= y1_abs + text_height.
                # The text's bottom edge is at (text_y + baseline). Must be <= y2_abs. So text_y <= y2_abs - baseline.
                text_y = max(y1_abs + text_height, min(ideal_text_y_baseline, y2_abs - baseline))

                # Put the masked text on the image
                cv2.putText(img, masked_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Convert BGR image to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Image with All Extracted Texts Annotated and Masked')
    plt.savefig('annotated_all_extracted_texts.png', dpi=300, bbox_inches='tight')
    # Save the figure to a file
    plt.show()


def annotate_targeted_texts(image_path: str, texts: list, draw_bbox: bool = True, fill_bbox_white: bool = False):
    # Load the original image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Define visual properties
    box_color = (0, 255, 0)  # Green in BGR
    text_color = (0, 0, 255) # Red in BGR
    white_color = (255, 255, 255) # White in BGR for filling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1


    # draw all boxes given in this list
    for (text_content, points) in texts:
        # Create masked text with asterisks
        masked_text = "*" * (len(text_content) // 2) + "*" * (len(text_content) % 2)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Calculate absolute coordinates by adding parent box offset
        x1_abs = int(min(x_coords))
        y1_abs = int(min(y_coords))
        x2_abs = int(max(x_coords))
        y2_abs = int(max(y_coords))

        # Ensure valid bounding box (min_x < max_x, min_y < max_y) before proceeding
        if x2_abs <= x1_abs or y2_abs <= y1_abs:
            print(f"Warning: Invalid bounding box for text '{text_content}'. Skipping.")
            continue

        # Fill the bounding box with white if the flag is true
        if fill_bbox_white:
            cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), white_color, cv2.FILLED)

        # Draw the bounding box outline if the flag is true
        if draw_bbox:
            cv2.rectangle(img, (x1_abs, y1_abs), (x2_abs, y2_abs), box_color, thickness)

        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(masked_text, font, font_scale, thickness)

        bbox_width = x2_abs - x1_abs
        bbox_height = y2_abs - y1_abs

        # Calculate centered horizontal position
        ideal_text_x = x1_abs + (bbox_width - text_width) // 2

        # Clamp text_x to ensure it stays within the horizontal bounds of the bbox
        text_x = max(x1_abs, min(ideal_text_x, x2_abs - text_width))

        # Calculate centered vertical position (baseline)
        # The overall vertical extent of the text (from top of ascenders to bottom of descenders) is text_height + baseline.
        # We want the center of this overall text extent to align with the center of the bounding box.
        bbox_center_y = y1_abs + bbox_height // 2

        # Calculate the ideal baseline y-coordinate such that the *vertical center* of the entire text string
        # (including ascenders and descenders) aligns with the center of the bounding box.
        ideal_text_y_baseline = bbox_center_y + (text_height - baseline) // 2

        # Clamp text_y to ensure it stays within vertical bounds
        # The text's top edge is at (text_y - text_height). Must be >= y1_abs. So text_y >= y1_abs + text_height.
        # The text's bottom edge is at (text_y + baseline). Must be <= y2_abs. So text_y <= y2_abs - baseline.
        text_y = max(y1_abs + text_height, min(ideal_text_y_baseline, y2_abs - baseline))

        # Put the masked text on the image
        cv2.putText(img, masked_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Convert BGR image to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('Image with All Extracted Texts Annotated and Masked')
    plt.savefig('annotated_targeted_texts.png', dpi=300, bbox_inches='tight')
    plt.show()

def mask_text_regions(
    image_path: str,
    texts: List[MaskedText],
    draw_bbox: bool = True,
    fill_bbox_white: bool = False,
) -> None:
    """
    Masks text regions in an image by drawing bounding boxes and
    overlaying masked text (asterisks).

    Parameters
    ----------
    image_path : str
        Path to the input image.

    texts : List[Tuple[str, BoundingBox]]
        List of tuples containing:
        - original text content
        - bounding box as [[x1, y1], [x2, y2]]

    draw_bbox : bool, default=True
        Whether to draw bounding box outlines.

    fill_bbox_white : bool, default=False
        Whether to fill bounding boxes with white color
        before drawing masked text.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")

    # Visual configuration
    box_color = (0, 255, 0)      # Green (BGR)
    text_color = (0, 0, 255)     # Red (BGR)
    fill_color = (255, 255, 255) # White
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 1

    for original_text, bbox in texts:
        masked_text = _generate_masked_text(original_text)

        x1, y1, x2, y2 = _normalize_bbox(bbox)
        if not _is_valid_bbox(x1, y1, x2, y2):
            print(f"Warning: Invalid bounding box for '{original_text}'. Skipping.")
            continue

        if fill_bbox_white:
            cv2.rectangle(image, (x1, y1), (x2, y2), fill_color, cv2.FILLED)

        if draw_bbox:
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        _draw_centered_text(
            image=image,
            text=masked_text,
            bbox=(x1, y1, x2, y2),
            font=font,
            font_scale=font_scale,
            color=text_color,
            thickness=thickness,
        )

    _display_image(image)
