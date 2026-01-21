import os
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

from utils.drawing import annotate_targeted_texts, draw_box_on_all_texts, mask_all_extracted_texts
from masking.masking_by_header import search_text_by_header
from masking.mask_all_match_text import search_by_matcher

class UIElementDetector:
  def __init__(self, model_path: str = "assets/best.pt"):
    self.model = YOLO(model_path)

  def predict(self, image_path: str):
    if self.model is None:
      raise RuntimeError("Model not loaded. Call load() first.")
    result = self.model.predict(image_path)

    extracted_data = []
    CLASS_NAMES = {
        0: "table_column",
        1: "text_field",
        2: "text_info"
    }

    if result[0].boxes is not None:
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, f"unknown_class_{cls_id}")
            extracted_data.append({
                "box": [x1, y1, x2, y2],
                "label": label
            })
    return extracted_data


class DataExtractor:
  def __init__(self, model_path: str):
    self.reader = easyocr.Reader(['en'])
    self.ui_element_extractor = UIElementDetector(model_path)

  def get_data_from_image(self, image_path: str):
    # Assign unique IDs to each extracted item and add placeholder for texts and header
    extracted_data = self.ui_element_extractor.predict(image_path)
    processed_data = []
    for i, item in enumerate(extracted_data):
        processed_item = {
            "uid": i + 1,  # Assigning a unique ID starting from 1
            "box": item["box"],
            "texts": [],  # Placeholder for extracted text
            "header": "",  # Placeholder for header information
            "type": item["label"] # Using the detected label as type
        }
        processed_data.append(processed_item)

    # print("Processed Data with Unique IDs:")
    # for item in processed_data:
    #     print(item)
    return processed_data

  def data_extraction_from_image(self, img_path: str):
    processed_data = self.get_data_from_image(img_path)
    # Load the original image once
    # Assuming the image path is the one used for prediction
    original_image = cv2.imread(img_path)

    if original_image is None:
        print(f"Error: Could not load image from {img_path}")
    else:
        # Iterate through processed_data and apply EasyOCR
        for item in processed_data:
            x1, y1, x2, y2 = map(int, item["box"])

            # Crop the image using bounding box coordinates
            cropped_image = original_image[y1:y2, x1:x2]

            # Check if the cropped image is valid
            if cropped_image.shape[0] > 0 and cropped_image.shape[1] > 0:
                # Apply EasyOCR to the cropped image
                ocr_results = self.reader.readtext(cropped_image)

                # Extract only the text from OCR results
                extracted_texts = []
                for (bbox, text, prob) in ocr_results:
                    # EasyOCR bounding box format is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    extracted_texts.append({"box": bbox, "text": text, "prob": prob})

                extracted_texts.sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))
                # # Apply sorting based on the item type
                # if item["type"] in ["text_info", "table_column"]:
                #     # Sort by y-coordinate (top-most first), then by x-coordinate (left-most first)
                #     extracted_texts.sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))
                # else:
                #     # Sort by x-coordinate (left-most first), then by y-coordinate (top-most first)
                #     extracted_texts.sort(key=lambda x: (x['box'][0][0], x['box'][0][1]))

                # Update the 'texts' field in processed_data
                item["texts"] = extracted_texts
            else:
                print(f"Warning: Cropped image for UID {item['uid']} is empty. Skipping OCR.")
                item["texts"] = [] # Ensure it's an empty list if cropping failed

        # make header
        for item in processed_data:
          if item["texts"]:
              # Assuming the first text in the sorted list is the header
              item["header"] = item["texts"][0]["text"]
          # print(item["header"])
    return processed_data


if __name__ == "__main__":
    model_path = "assets/best.pt"
    data_extractor = DataExtractor(model_path)
    img_path = "src/images/ss-1.jpeg"
    # img_path = "/content/drive/MyDrive/POC-table-masking/images/2.JPG"
    processed_data = data_extractor.data_extraction_from_image(img_path)
    draw_box_on_all_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=False)
    mask_all_extracted_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=True)
    
    
    # masking by header
    header_text = "Description"
    texts = search_text_by_header(processed_data, header_text)
    annotate_targeted_texts(img_path, texts, True, True)
    
    # masking by matcher
    texts = search_by_matcher(processed_data, "Touring Bike")
    annotate_targeted_texts(img_path, texts, True, True)
    
    