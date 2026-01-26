# load train yolo model and predict on image
import os
import cv2
import numpy as np
import pytesseract
from .data_extractor import UIElementDetector

class TessaractDataExtractor:
    def __init__(self, model_path: str):
        self.tesseract_config = "--oem 3 --psm 6"
        self.ui_element_extractor = UIElementDetector(model_path)

    def get_data_from_image(self, image_path: str):
        extracted_data = self.ui_element_extractor.predict(image_path)
        processed_data = []

        for i, item in enumerate(extracted_data):
            processed_item = {
                "uid": i + 1,
                "box": item["box"],        # [x1, y1, x2, y2]
                "texts": [],
                "header": "",
                "type": item["label"]
            }
            processed_data.append(processed_item)

        return processed_data

    def data_extraction_from_image(self, img_path: str):
        processed_data = self.get_data_from_image(img_path)
        original_image = cv2.imread(img_path)

        if original_image is None:
            print(f"Error: Could not load image from {img_path}")
            return processed_data

        # =============================
        # APPLY PYTESSERACT OCR
        # =============================
        for item in processed_data:
            x1, y1, x2, y2 = map(int, item["box"])
            cropped_image = original_image[y1:y2, x1:x2]

            if cropped_image.shape[0] <= 0 or cropped_image.shape[1] <= 0:
                print(f"Warning: Cropped image for UID {item['uid']} is empty.")
                item["texts"] = []
                continue

            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            data = pytesseract.image_to_data(
                gray,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )

            extracted_texts = []

            for i in range(len(data["text"])):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])

                if not text or conf < 0:
                    continue

                x = int(data["left"][i])
                y = int(data["top"][i])
                w = int(data["width"][i])
                h = int(data["height"][i])

                # ✅ Convert Tesseract bbox → EasyOCR-like bbox
                bbox = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]

                extracted_texts.append({
                    "box": bbox,
                    "text": text,
                    "prob": conf / 100.0
                })

            # =============================
            # SORT TEXTS (Top → Bottom, Left → Right)
            # =============================
            extracted_texts.sort(
                key=lambda x: (x["box"][0][1], x["box"][0][0])
            )

            item["texts"] = extracted_texts

        # =============================
        # HEADER SELECTION
        # =============================
        for item in processed_data:
            if item["texts"]:
                item["header"] = item["texts"][0]["text"]

        return processed_data
