import sys
import os
import cv2
import json
import argparse
from common.logger import Logger


from data_extraction.data_extraction_tessaract import TessaractDataExtractor
from data_extraction.data_extractor_easyocr import EasyOcrDataExtractor
from utils.table_data_handling import process_table_data
from utils.helper import remove_unnecessary_characters, save_ocr_data
from utils.merge_texts import box_stats, merge_texts
from utils.merge_ocr_data import merge_ocr_data
from utils.drawing import annotate_targeted_texts, draw_box_on_all_texts, mask_all_extracted_texts
from masking.masking_by_header import search_text_by_header

logger = Logger.get_logger("main")

def header_selection(processed_data, image_path):
    """Select header from extracted texts and assign to 'header' key."""
    for item in processed_data:
        if item["texts"]:
            # sort based on logic
            if item["type"] == "text_field":
                # left to right sort
                item["texts"].sort(key=lambda x: box_stats(x["box"])["x_min"])
            elif item["type"] == "top_down_text_field":
                # top to bottom sort
                item["texts"].sort(key=lambda x: box_stats(x["box"])["y_min"])
            elif item["type"] == "table_column":
                # we will take help in morphological operation in that case
                header, text_boxes = process_table_data(item["texts"], item["box"], image_path)
                item["texts"] = text_boxes
                
            try:    
                item["header"] = remove_unnecessary_characters(item["texts"][0]["text"])
                # print(f" Header: {item['header']} for type: {item['type']}")
                item["texts"] = item["texts"][1:]
            except Exception as e:
                print(f"Error {e} while processing Image: {image_path}")
            
    return processed_data

def intermediate_drawing(img_path, processed_data, output_dir = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    """Draw intermediate annotated images for visualization."""
    annotated1 = draw_box_on_all_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=False)
    annotated2 = mask_all_extracted_texts(img_path, processed_data, draw_bbox=True, fill_bbox_white=True)
    # write annotated images
    image_name = img_path.split("/")[-1].split(".")[0]
    cv2.imwrite(os.path.join(output_dir, f"annotated_all_texts_{image_name}.png"), cv2.cvtColor(annotated1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"masked_all_texts_{image_name}.png"), cv2.cvtColor(annotated2, cv2.COLOR_RGB2BGR))


def masking_by_header(header_texts: list, processed_data, img_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    """Mask texts based on multiple headers."""
    # multiple headers may be provided, mask each one by one and show in single image
    texts_to_annotate = []
    is_found_any_header = False
    for header_text in header_texts:
        texts, match_result = search_text_by_header(processed_data, header_text, match_threshold=0.95)
        texts_to_annotate.extend(texts)
        if match_result:
            is_found_any_header = True

    if not texts_to_annotate:
        logger.warning(f"Sorry! No texts found for headers: {header_texts}")
        # # mask all texts if no header is found, this is optional, you can just skip masking if no header is found
        # annotated = mask_all_extracted_texts(img_path, processed_data, draw_bbox=False, fill_bbox_white=True)
        # img_name = img_path.split("/")[-1].split(".")[0]
        # cv2.imwrite(os.path.join(output_dir, f"masked_by_headers_{img_name}.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        # if not is_found_any_header:
        all_texts = TessaractDataExtractor.texts_extraction_from_image(img_path)
        annotated = annotate_targeted_texts(img_path, all_texts, draw_bbox=False, fill_bbox_white=True)
        img_name = img_path.split("/")[-1].split(".")[0]
        cv2.imwrite(os.path.join(output_dir, f"masked_by_headers_{img_name}.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            
        return
    
    annotated = annotate_targeted_texts(img_path, texts_to_annotate, draw_bbox=True, fill_bbox_white=True)
    # save masking image
    img_name = img_path.split("/")[-1].split(".")[0]
    cv2.imwrite(os.path.join(output_dir, f"masked_by_headers_{img_name}.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

def main(headers_text: list, image_path: str, model_path: str):
    # Initialize OCR Data Extractors
    easy_ocr_dex = EasyOcrDataExtractor(model_path)
    tessaract_ocr_dex = TessaractDataExtractor(model_path)

    # make outputs directory
    image_name = image_path.split("/")[-1].split(".")[0]
    output_dir = f"outputs/{image_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Step-1: Extract data from both OCRs
    # easy_ocr_data = easy_ocr_dex.data_extraction_from_image(image_path)
    tessaract_ocr_data = tessaract_ocr_dex.data_extraction_from_image(image_path)

    # Step-2: Merge Two OCR data
    # processed_data = merge_ocr_data(easy_ocr_data, tessaract_ocr_data, iou_threshold=0.3)
    
    # merge horizontal texts and vertical texts with small gap
    processed_data = merge_texts(tessaract_ocr_data)
    # Now select header
    processed_data = header_selection(processed_data, image_path)

    # Save extracted text with headers
    save_ocr_data(image_path, processed_data, output_dir)
    # This drawing is optional, just for visualization
    intermediate_drawing(image_path, processed_data, output_dir)
    
    # Step-4: Targeted masking
    masking_by_header(headers_text, processed_data, image_path, output_dir)

    # # masking by matcher
    # texts = search_by_matcher(processed_data, "Touring Bike")
    # annotate_targeted_texts(image_path, texts, True, True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="UI Element–Aware Data Masking PoC"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input UI image"
    )

    parser.add_argument(
        "--headers",
        nargs="+",
        required=True,
        help="Table column headers to mask (space separated)"
    )

    parser.add_argument(
        "--model",
        default="assets/best.pt",
        help="Path to YOLO UI detection model (default: assets/best.pt)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        logger.error(
            f"❌ Model not found at {args.model}\n"
            f"Please download the YOLO model and place it inside the assets/ folder."
        )
        sys.exit(1)

    # Validate image path
    if not os.path.exists(args.image):
        logger.error(f"❌ Image not found: {args.image}")
        sys.exit(1)

    main(
        headers_text=args.headers,
        image_path=args.image,
        model_path=args.model
    )
