import easyocr
import numpy as np
from typing import List, Tuple

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def process_image(self, image: np.ndarray) -> List[Tuple]:
        results = self.reader.readtext(image)
        return [(res[0], res[1], res[2]) for res in results]
    
        
    def merge_ocr_results(self, ocr_results: List[Tuple], distance_threshold: float = 20):
        """
        Merge OCR bounding boxes that overlap or are close to each other.

        Args:
            ocr_results: List of (bbox, text, confidence) tuples from EasyOCR
            distance_threshold: Maximum distance to consider boxes as mergeable

        Returns:
            List of merged (bbox, text, confidence) tuples
        """
        if not ocr_results:
            return []

        # Convert bboxes to rectangles for easier processing
        boxes = []
        for bbox, text, conf in ocr_results:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            boxes.append({
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'text': text, 'conf': conf,
                'merged': False
            })

        # Sort boxes by y position (top to bottom), then x position (left to right)
        boxes.sort(key=lambda b: (b['y_min'], b['x_min']))

        merged_results = []

        for i, box in enumerate(boxes):
            if box['merged']:
                continue

            # Start a new merged group
            merged_group = [box]
            box['merged'] = True

            # Try to merge with subsequent boxes
            for j in range(i + 1, len(boxes)):
                other = boxes[j]
                if other['merged']:
                    continue

                # Get the bounding box of current merged group
                group_x_min = min(b['x_min'] for b in merged_group)
                group_x_max = max(b['x_max'] for b in merged_group)
                group_y_min = min(b['y_min'] for b in merged_group)
                group_y_max = max(b['y_max'] for b in merged_group)

                # Check if boxes should be merged
                # 1. Check horizontal overlap/proximity
                horizontal_gap = max(other['x_min'], group_x_min) - min(other['x_max'], group_x_max)
                horizontal_overlap = horizontal_gap < distance_threshold

                # 2. Check vertical overlap/proximity
                vertical_gap = max(other['y_min'], group_y_min) - min(other['y_max'], group_y_max)
                vertical_overlap = vertical_gap < distance_threshold

                # 3. Check if they're on roughly the same line (y-overlap)
                y_center_group = (group_y_min + group_y_max) / 2
                y_center_other = (other['y_min'] + other['y_max']) / 2
                same_line = abs(y_center_group - y_center_other) < distance_threshold

                if (horizontal_overlap and vertical_overlap) or (same_line and horizontal_overlap):
                    merged_group.append(other)
                    other['merged'] = True

            # Create merged bounding box and text
            merged_x_min = min(b['x_min'] for b in merged_group)
            merged_x_max = max(b['x_max'] for b in merged_group)
            merged_y_min = min(b['y_min'] for b in merged_group)
            merged_y_max = max(b['y_max'] for b in merged_group)

            # Sort texts by y position (top to bottom), then x position (left to right)
            merged_group.sort(key=lambda b: (b['y_min'], b['x_min']))
            merged_text = ' '.join(b['text'] for b in merged_group)
            merged_conf = sum(b['conf'] for b in merged_group) / len(merged_group)

            # Convert back to bbox format (4 corners)
            merged_bbox = [
                [merged_x_min, merged_y_min],
                [merged_x_max, merged_y_min],
                [merged_x_max, merged_y_max],
                [merged_x_min, merged_y_max]
            ]

            merged_results.append((merged_bbox, merged_text, merged_conf))

            # Print debug info for merged boxes
            if len(merged_group) > 1:
                original_texts = [b['text'] for b in merged_group]
                print(f"📦 MERGED: {original_texts} → '{merged_text}'")
                print(f"   Box: ({int(merged_x_min)}, {int(merged_y_min)}) to ({int(merged_x_max)}, {int(merged_y_max)})")

        print(f"\n✨ OCR Merging: {len(ocr_results)} boxes → {len(merged_results)} boxes")
        return merged_results