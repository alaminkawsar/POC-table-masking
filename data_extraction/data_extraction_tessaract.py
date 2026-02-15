# load train yolo model and predict on image
import os
import cv2
import numpy as np
from detection.ui_detector import UIElementDetector
import pytesseract

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

    @staticmethod
    def horizontal_iou(box1, box2):
        # box = (x1, y1, x2, y2)
        x1, _, x2, _ = box1
        x1b, _, x2b, _ = box2

        inter = max(0, min(x2, x2b) - max(x1, x1b))
        union = max(x2, x2b) - min(x1, x1b)

        return inter / union if union > 0 else 0


    # -----------------------------
    # Merge vertical aligned texts
    # -----------------------------

    def merge_vertical_texts(
        self,
        texts,
        vertical_gap_threshold=12,
        horizontal_overlap_threshold=0.6
    ):
        if not texts:
            return []

        items = []
        for t in texts:
            x1, y1, x2, y2 = self.box_to_xyxy(t["box"])
            items.append({
                "orig": t,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

        # Sort top → bottom
        items.sort(key=lambda i: i["y1"])

        merged_blocks = []

        for item in items:
            if not merged_blocks:
                merged_blocks.append([item])
                continue

            last_block = merged_blocks[-1][-1]

            vertical_gap = item["y1"] - last_block["y2"]
            h_iou = self.horizontal_iou(
                (item["x1"], item["y1"], item["x2"], item["y2"]),
                (last_block["x1"], last_block["y1"], last_block["x2"], last_block["y2"])
            )

            if vertical_gap <= vertical_gap_threshold and h_iou >= horizontal_overlap_threshold:
                merged_blocks[-1].append(item)
            else:
                merged_blocks.append([item])

        merged_texts = []

        for block in merged_blocks:
            x1 = min(i["x1"] for i in block)
            y1 = min(i["y1"] for i in block)
            x2 = max(i["x2"] for i in block)
            y2 = max(i["y2"] for i in block)

            text = " ".join(i["orig"]["text"] for i in block)
            prob = max(i["orig"]["prob"] for i in block)

            merged_texts.append({
                "box": self.xyxy_to_box(x1, y1, x2, y2),
                "text": text,
                "prob": float(prob)
            })

        return merged_texts


    # -----------------------------
    # Merge horizontally aligned texts
    # -----------------------------
    def merge_horizontal_texts(
      self,
      texts,
      h_type,
      line_threshold=8,
      max_gap_ratio=1.5
    ):
      """
      Merge horizontally aligned OCR texts into line-level boxes.

      Rules:
      - Texts must be vertically aligned (same line)
      - Horizontal gap must be small relative to character width
      """

      if not texts:
          return []

      # -------- Step 1: Normalize boxes --------
      items = []
      for t in texts:
          x1, y1, x2, y2 = self.box_to_xyxy(t["box"])
          y_center = (y1 + y2) / 2

          items.append({
              "orig": t,
              "x1": x1,
              "y1": y1,
              "x2": x2,
              "y2": y2,
              "y_center": y_center
          })

      if h_type == "text_field":
        # Sort (left → right)
        items.sort(key=lambda i: i["x1"])
      else:
        # Sort (top → bottom)
        # Sort (top → bottom)
        items.sort(key=lambda i: i["y_center"]) 

      lines = []

      # -------- Step 3: Line grouping with gap check --------
      for item in items:
          if not lines:
              lines.append([item])
              continue

          last_line = lines[-1]
          last_item = last_line[-1]

          # Vertical alignment check
          same_line = abs(item["y_center"] - last_item["y_center"]) <= line_threshold

          if same_line:
              # Horizontal gap check
              horizontal_gap = item["x1"] - last_item["x2"]

              last_width = last_item["x2"] - last_item["x1"]
              char_width = last_width / max(len(last_item["orig"]["text"]), 1)

              if horizontal_gap <= char_width * max_gap_ratio:
                  last_line.append(item)
              else:
                  lines.append([item])
          else:
              lines.append([item])

      # -------- Step 4: Merge boxes & texts --------
      merged_texts = []

      for line in lines:
          line.sort(key=lambda i: i["x1"])

          x1 = min(i["x1"] for i in line)
          y1 = min(i["y1"] for i in line)
          x2 = max(i["x2"] for i in line)
          y2 = max(i["y2"] for i in line)

          # Smart space insertion
          text_parts = []
          for i, curr in enumerate(line):
              if i > 0:
                  prev = line[i - 1]
                  gap = curr["x1"] - prev["x2"]

                  prev_char_width = (
                      (prev["x2"] - prev["x1"]) /
                      max(len(prev["orig"]["text"]), 1)
                  )

                  if gap > prev_char_width:
                      text_parts.append(" ")

              text_parts.append(curr["orig"]["text"])

          text = "".join(text_parts)

          # Probability aggregation (length-weighted)
          prob = sum(
              len(i["orig"]["text"]) * i["orig"]["prob"]
              for i in line
          ) / sum(len(i["orig"]["text"]) for i in line)

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
        
        print("Performing Tesseract OCR on detected UI elements...\n")
        
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
                # print(f"pytessarct text: {text}")
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
            h_type = item["type"]

            # 🔗 MERGE HORIZONTAL LINES HERE
            line_texts = self.merge_horizontal_texts(raw_texts, h_type)
            # final_texts = self.merge_vertical_texts(line_texts)
            item["texts"] = line_texts
        # Sort the each texts to find the header
        # -------------------------------------
        
        for item in processed_data:
            texts = item.get("texts", [])
            if not texts:
                continue

            if item["type"] == "text_field":
                # Left → Right
                item["texts"] = sorted(
                    texts,
                    key=lambda t: self.box_to_xyxy(t["box"])[0]  # x1
                )
            else:
                # Top → Bottom
                item["texts"] = sorted(
                    texts,
                    key=lambda t: self.box_to_xyxy(t["box"])[1]  # y1
                )

        # Header selection
        for item in processed_data:
            if item["texts"]:
                item["header"] = item["texts"][0]["text"]
                
                # skip ':' character from header
                if item["header"].endswith(":"):
                    item["header"] = item["header"][:-1].strip()
                    
                item["texts"] = item["texts"][1:]
                
           
        return processed_data
