# load train yolo model and predict on image
import os
import cv2
import numpy as np
import pytesseract
from .data_extractor_easyocr import UIElementDetector



class TessaractDataExtractor:
    def __init__(self, model_path: str):
        self.tesseract_config = "--oem 3 --psm 6"
        self.ui_element_extractor = UIElementDetector(model_path)

    # -----------------------------
    # Helper: box utils
    # -----------------------------
    @staticmethod
    def box_to_xyxy(box):
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def xyxy_to_box(x1, y1, x2, y2):
        return [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]

    # -----------------------------
    # Merge horizontally aligned texts
    # -----------------------------
    def merge_horizontal_texts(self, texts, line_threshold=8):
        if not texts:
            return []

        items = []
        for t in texts:
            x1, y1, x2, y2 = self.box_to_xyxy(t["box"])
            y_center = (y1 + y2) / 2

            items.append({
                "orig": t,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "y_center": y_center
            })

        # Sort top → bottom, left → right
        items.sort(key=lambda i: (i["y_center"], i["x1"]))

        lines = []

        for item in items:
            if not lines:
                lines.append([item])
                continue

            last_line = lines[-1]
            last_y = last_line[0]["y_center"]

            if abs(item["y_center"] - last_y) <= line_threshold:
                last_line.append(item)
            else:
                lines.append([item])

        merged_texts = []

        for line in lines:
            line.sort(key=lambda i: i["x1"])

            x1 = min(i["x1"] for i in line)
            y1 = min(i["y1"] for i in line)
            x2 = max(i["x2"] for i in line)
            y2 = max(i["y2"] for i in line)

            text = " ".join(i["orig"]["text"] for i in line)
            prob = max(i["orig"]["prob"] for i in line)

            merged_texts.append({
                "box": self.xyxy_to_box(x1, y1, x2, y2),
                "text": text,
                "prob": float(prob)
            })

        return merged_texts

    # -----------------------------
    # UI detection
    # -----------------------------
    def get_data_from_image(self, image_path: str):
        extracted_data = self.ui_element_extractor.predict(image_path)
        processed_data = []

        for i, item in enumerate(extracted_data):
            processed_data.append({
                "uid": i + 1,
                "box": item["box"],        # [x1, y1, x2, y2]
                "texts": [],
                "header": "",
                "type": item["label"]
            })

        return processed_data

    # -----------------------------
    # Main OCR pipeline
    # -----------------------------
    def data_extraction_from_image(self, img_path: str):
        processed_data = self.get_data_from_image(img_path)
        original_image = cv2.imread(img_path)

        if original_image is None:
            print(f"Error: Could not load image from {img_path}")
            return processed_data

        for item in processed_data:
            x1, y1, x2, y2 = map(int, item["box"])
            cropped_image = original_image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                item["texts"] = []
                continue

            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            data = pytesseract.image_to_data(
                gray,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            raw_texts = []

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if not text or conf < 0:
                    continue

                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                raw_texts.append({
                    "box": [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ],
                    "text": text,
                    "prob": conf / 100.0
                })

            # 🔗 MERGE HORIZONTAL LINES HERE
            item["texts"] = self.merge_horizontal_texts(raw_texts)

        # Header selection
        for item in processed_data:
            if item["texts"]:
                item["header"] = item["texts"][0]["text"]

        return processed_data
