import os
import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from data_extraction.data_extractor import DataExtractor
from masking.masking import mask_texts_from_ocr_data
from ocr.easy_ocr import EasyOCR

def mask_text_in_image(image_path: str, texts_to_mask: list, output_path: str):
    """
    Masks specified texts in the image by drawing white rectangles over their bounding boxes.

    :param image_path: Path to the input image.
    :param texts_to_mask: List of tuples containing text and its bounding box.
                          Each tuple is of the form (text, [[x1, y1], [x2, y2]]).
    :param output_path: Path to save the output masked image.
    """
    # Load the image
    image = cv2.imread(image_path)

    for text, box in texts_to_mask:
        # box format: [[x1, y1], [x2, y2]]
        x1, y1 = box[0]
        x2, y2 = box[1]

        # Draw a white rectangle over the text area
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    # Save the masked image
    cv2.imwrite(output_path, image)
    

    
def main():
    
    model_path = "assets/best.pt"
    ocr_engine = EasyOCR()
    data_extractor = DataExtractor(model_path, ocr_engine)
    img_path = "src/images/ss-1.jpeg"
    processed_data = data_extractor.data_extraction_from_image(img_path)
    # annotate_all_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=False)
    # annotate_all_extracted_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=True)

    header_text = "Ship To"
    masked_image = mask_texts_from_ocr_data(
        img_path,
        processed_data,
        mode="by_header",
        header_name=header_text,
    )
    cv2.imwrite("masked_by_header.png", masked_image)



if __name__ == "__main__":
    main()
