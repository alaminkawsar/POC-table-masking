# extract all boxes related to user target text

# text to be masked
# targeted_text = "Touring Bike"

def search_by_matcher(processed_data, targeted_text):
    texts = []

    # first mask all text for the given header
    for item in processed_data:
            parent_x1, parent_y1, _, _ = item["box"]
            texts_to_annotate = item["texts"]

            for text_info in texts_to_annotate:
                if "text" in text_info and "box" in text_info:
                    text_content = text_info["text"]
                    # match with targeted text
                    if text_content.strip() != targeted_text.strip():
                        continue

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
                    texts.append((text_content, [[x1_abs, y1_abs], [x2_abs, y2_abs]]))
                    
    return texts
