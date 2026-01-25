import copy

from utils.helper import compute_iou, easyocr_bbox_to_xyxy

def merge_ocr_data(easy_ocr_data: list, tessaract_ocr_data: list, iou_threshold: float = 0.3) -> list:
    """Merges text detections from EasyOCR and Tesseract OCR, adding Tesseract detections
    that are not sufficiently covered by EasyOCR detections.

    Args:
        easy_ocr_data (list): List of processed data from EasyOCR.
        tessaract_ocr_data (list): List of processed data from Tesseract OCR.
        iou_threshold (float): IOU threshold to determine if a Tesseract detection
                                is already covered by an EasyOCR detection.

    Returns:
        list: Merged OCR data.
    """
    # Create a deep copy of easy_ocr_data to store the merged results
    merged_data = copy.deepcopy(easy_ocr_data)

    # Convert Tesseract data to a dictionary for easier lookup by uid
    tessaract_data_map = {item['uid']: item for item in tessaract_ocr_data}

    # Iterate through each item in the copied easy_ocr_data
    for easy_ocr_item in merged_data:
        uid = easy_ocr_item['uid']

        # Find the corresponding tessaract_ocr_item
        tessaract_ocr_item = tessaract_data_map.get(uid)
        if tessaract_ocr_item is None:
            continue # Skip if no matching Tesseract item

        # Extract existing texts from easy_ocr_item and convert their bounding boxes to xyxy format
        easy_ocr_boxes_xyxy = [
            easyocr_bbox_to_xyxy(text_info['box']) 
            for text_info in easy_ocr_item['texts']
        ]

        # For each text entry in tessaract_ocr_item['texts']
        for tess_text_info in tessaract_ocr_item['texts']:
            # Convert its bounding box to xyxy format
            tess_box_xyxy = easyocr_bbox_to_xyxy(tess_text_info['box'])

            is_duplicate = False
            # Iterate through EasyOCR bounding boxes and compute IOU
            for easy_box_xyxy in easy_ocr_boxes_xyxy:
                if compute_iou(tess_box_xyxy, easy_box_xyxy) > iou_threshold:
                    is_duplicate = True
                    break

            # If not a duplicate, append the Tesseract text entry
            if not is_duplicate:
                easy_ocr_item['texts'].append(tess_text_info)

        # Sort the easy_ocr_item['texts'] list after merging
        easy_ocr_item['texts'].sort(key=lambda x: (x['box'][0][1], x['box'][0][0]))

    return merged_data

print("Function `merge_ocr_data` defined.")