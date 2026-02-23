import os
import numpy as np
import cv2

def xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)

def intersection_xyxy(boxA, boxB):
    """
    boxA, boxB: (x_min, y_min, x_max, y_max)
    returns: (x_min, y_min, x_max, y_max) or None
    """
    x_left   = max(boxA[0], boxB[0])
    y_top    = max(boxA[1], boxB[1])
    x_right  = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right <= x_left or y_bottom <= y_top:
        return None  # no overlap

    return [int(x_left), int(y_top), int(x_right), int(y_bottom)]


def points_to_rect(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return (x_min, y_min, x_max - x_min, y_max - y_min)

def rect_intersection(box1, box2):
    x1, y1, w1, h1 = np.array(box1, dtype=np.int32)
    x2, y2, w2, h2 = np.array(box2, dtype=np.int32)

    x_left   = max(x1, x2)
    y_top    = max(y1, y2)
    x_right  = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right <= x_left or y_bottom <= y_top:
        return None  # no overlap

    return (
        x_left,
        y_top,
        x_right - x_left,
        y_bottom - y_top
    )
    
def refine_mask_with_morphology(mask, output_path=None):
    """
    Refine mask using morphological operation to better separate words.
    
    Args:
        mask_path (str): Path to the binary mask image.
        output_path (str, optional): Path to save the refined mask. If None, it won't be saved.
    Returns:
        np.ndarray: Refined binary mask.
    """
    
    # Invert if mas has white bacground (text is white on black is better for connected components)
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)
        # print("Mask inverted: background was white, now black")
    # Apply morphological opening to remove noise
    kernel_open = np.ones((2,2), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # save refined mask
    if output_path:
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, mask_opened)

    return mask_opened

def get_word_bounding_boxes_from_image(image_path, min_area=20, max_area=50000):
    """
    Get bounding boxes of words from the binary mask.
    
    Args:
        mask_path (str): Path to the binary mask image.
        min_area (int): Minimum area of connected component to be considered a word.
        max_area (int): Maximum area of connected component to be considered a word.
    Returns:
        List[Tuple[int, int, int, int]]: List of bounding boxes (x, y, w, h) for each detected word.
    """
    # Read original image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step-1: Apply erosion to separate connected words
    kernel = np.ones((4, 4), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=3)
    
    # Step-2: Refine mask with morphological operations
    refined_mask = refine_mask_with_morphology(img_erosion)
    
    # Step-3: Make Binary Image and Find Connected Components
    # make binary image on refined_mask
    _, refined_mask_binary = cv2.threshold(refined_mask, 90, 255, cv2.THRESH_BINARY)
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(refined_mask_binary, connectivity=8)

    FILL_THRESHOLD = 0.25   # tune this (0.9 = very strict)

    boxes = []

    for i in range(1, num_labels):  # skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        bbox_area = w * h
        fill_ratio = area / bbox_area

        if fill_ratio >= FILL_THRESHOLD:
            boxes.append((x, y, w, h))

    return img_erosion, refined_mask, refined_mask_binary, boxes


def process_table_data(texts: list, det_box, image_path: str):
    """
    Process table data to extract headers and word bounding boxes.
    
    Args:
        processed_data (list): List of detected UI elements with their texts and bounding boxes.
        image_path (str): Path to the input image for morphological operations.
    """
    img_erosion, refined_mask, refined_mask_binary, word_boxes = get_word_bounding_boxes_from_image(image_path)

    # Initialize lists to hold headers and word boxes
    headers = []
    new_word_boxes = []    
    for box1 in word_boxes:
        # convert box1 → xyxy
        box1_xyxy = xywh_to_xyxy(box1)
        # find intersection
        inter = intersection_xyxy(box1_xyxy, det_box)
        if not inter:
            continue  # Skip if no intersection with the detected table column box
        
        new_word_boxes.append(inter)  # Keep the original xywh format for word boxes
    
    # sort new word boxes top to bottom
    new_word_boxes.sort(key=lambda b: b[1])  # Sort by y_min
    # print("New word boxes after intersection and sorting:", len(new_word_boxes))
    first_box = new_word_boxes[0] if new_word_boxes else None
    
    # Debuging purposes
    # if first_box:
    #     print("First box (potential header box):", first_box)
    #     draw_single_box(image_path, first_box, "outputs/first_box.png")
        
    # now take the first box from new_word_boxes and find how much texts are intersecting with that box, those will be the headers
    # sort texts top to bottom
    texts.sort(key=lambda t: t["box"][0][1])  # Sort by y_min
    headers = []
    removed_indices = []

    if new_word_boxes:
        first_box = new_word_boxes[0]
        # trace index which will remove later
        for i in range(len(texts)):
            text = texts[i]
            # print("Checking text:", text["text"], "with box:", text["box"])
            text_box = text["box"]  #text["box"] 4 corners points
            # print("Text box points:", text_box)
            
            parent_x1, parent_y1, parent_x2, parent_y2 = det_box
            x_coords = [p[0] for p in text_box]
            y_coords = [p[1] for p in text_box]
            # Calculate absolute coordinates by adding parent box offset
            x1_abs = int(parent_x1 + min(x_coords))
            y1_abs = int(parent_y1 + min(y_coords))
            x2_abs = int(parent_x1 + max(x_coords))
            y2_abs = int(parent_y1 + max(y_coords))
            text_box_xyxy = (x1_abs, y1_abs, x2_abs, y2_abs) # now it is (x_min, y_min, x_max, y_max)
            # text_box_xyxy = xywh_to_xyxy(text_box_xyxy) # now it is (x_min, y_min, x_max, y_max)
            # print("Text box in xyxy format:", text_box_xyxy, "first box:", first_box)
            inter = intersection_xyxy(text_box_xyxy, first_box)
            if inter:
                # print("Found header:", text["text"])
                headers.append(text["text"])
                # remove the header text from texts list, so that it won't be considered in word boxes merging
                removed_indices.append(i)

    # join headers with space sequentially
    headers = " ".join(headers)
    # print("Final header after merging:", headers)

    # remove indices
    for index in sorted(removed_indices, reverse=True):
        del texts[index]
    # merge word boxes with given text boxes if they have intersection,
    # otherwise keep the word boxes
    merged_word_boxes = []
    # remove first one from new_word_boxes because it is header box, we will add it later with header text
    for word_box in new_word_boxes:
        merged = False
        for text in texts:
            
            text_box = text["box"]  #text["box"] 4 corners points
            parent_x1, parent_y1, parent_x2, parent_y2 = det_box
            x_coords = [p[0] for p in text_box]
            y_coords = [p[1] for p in text_box]
            # Calculate absolute coordinates by adding parent box offset
            x1_abs = int(parent_x1 + min(x_coords))
            y1_abs = int(parent_y1 + min(y_coords))
            x2_abs = int(parent_x1 + max(x_coords))
            y2_abs = int(parent_y1 + max(y_coords))
            text_box_xyxy = (x1_abs, y1_abs, x2_abs, y2_abs) # now it is (x_min, y_min, x_max, y_max)
            
            # print("Merging word box:", word_box, "with text box:", text_box_xyxy)
            inter = intersection_xyxy(word_box, text_box_xyxy)
            if inter:
                # print("Merging word box:", word_box, "with text box:", text_box_xyxy, "Text:", text["text"])
                # subtract parent box offset from text_box_xyxy to get relative coordinates
                # and make 4 corner points [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                text_box_relative = [
                    (text_box_xyxy[0] - parent_x1, text_box_xyxy[1] - parent_y1),
                    (text_box_xyxy[2] - parent_x1, text_box_xyxy[1] - parent_y1),
                    (text_box_xyxy[2] - parent_x1, text_box_xyxy[3] - parent_y1),
                    (text_box_xyxy[0] - parent_x1, text_box_xyxy[3] - parent_y1)
                ]
                merged_word_boxes.append({
                    "text": text["text"],
                    "box": text_box_relative,
                    "prob": text.get("prob", 0.0)  # Use text's prob if available, otherwise default to 1.0
                })
                merged = True
                break
        
        # if the word box is not merged with any text box, keep it as it is with empty text and 0.0 prob
        # subtract parent box offset from word_box to get relative coordinates
        if not merged:
            word_box_relative = [
                (word_box[0] - parent_x1, word_box[1] - parent_y1),
                (word_box[2] - parent_x1, word_box[1] - parent_y1),
                (word_box[2] - parent_x1, word_box[3] - parent_y1),
                (word_box[0] - parent_x1, word_box[3] - parent_y1)
            ]
            merged_word_boxes.append({
                "text": "***",
                "box": word_box_relative,
                "prob": 0.0
            })
    # put headers in the first box of merged_word_boxes
    # replace first one with header text and first box
    if headers:
        # subtract parent box offset from first_box to get relative coordinates
        first_box_relative = [
            (new_word_boxes[0][0] - parent_x1, new_word_boxes[0][1] - parent_y1),
            (new_word_boxes[0][2] - parent_x1, new_word_boxes[0][1] - parent_y1),
            (new_word_boxes[0][2] - parent_x1, new_word_boxes[0][3] - parent_y1),
            (new_word_boxes[0][0] - parent_x1, new_word_boxes[0][3] - parent_y1)
        ]
        merged_word_boxes[0] = {
            "text": headers,
            "box": first_box_relative,
            "prob": 1.0
        }
        
    return headers, merged_word_boxes

def draw_table_data(image_path, word_boxes, output_path):
    img = cv2.imread(image_path)
    for box in word_boxes:
        x_min, y_min, w, h = box["box"]
        cv2.rectangle(img, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
        cv2.putText(img, box["text"], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, img)
    
    
def draw_single_box(image_path, box, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = cv2.imread("/home/kawsar/Desktop/Image-Masking/POC-table-masking/outputs/first_box.png")
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    # if output path exist then add number on the box
    cv2.imwrite(output_path, img)