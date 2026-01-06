import cv2
import easyocr
import numpy as np
from typing import List, Tuple, Dict


def merge_ocr_results(ocr_results: List[Tuple], distance_threshold: float = 20):
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


def clip_and_split_ocr_boxes(merged_ocr: List[Tuple], columns: List[Dict]) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Clip and split OCR bounding boxes to prevent overflow across column boundaries.
    Also identify text boxes that are completely outside all columns.

    Args:
        merged_ocr: List of (bbox, text, confidence) tuples
        columns: List of column dictionaries with boundaries

    Returns:
        Tuple of (clipped_results, outside_results):
        - clipped_results: List of (bbox, text, confidence, column_id) tuples for text inside columns
        - outside_results: List of (bbox, text, confidence) tuples for text outside all columns
    """
    clipped_results = []
    outside_results = []
    split_count = 0
    clipped_count = 0

    for (bbox, text, conf) in merged_ocr:
        # Get bounding rectangle
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        bb_x_min, bb_x_max = min(x_coords), max(x_coords)
        bb_y_min, bb_y_max = min(y_coords), max(y_coords)

        # Calculate center to determine primary column
        cx = (bb_x_min + bb_x_max) / 2
        cy = (bb_y_min + bb_y_max) / 2

        # Find all columns this box overlaps with
        overlapping_columns = []
        for col in columns:
            # Check if box overlaps with column
            x_overlap = not (bb_x_max < col['x_start'] or bb_x_min > col['x_end'])
            y_overlap = not (bb_y_max < col['y_start'] or bb_y_min > col['y_end'])

            if x_overlap and y_overlap:
                # Calculate overlap area
                overlap_x_min = max(bb_x_min, col['x_start'])
                overlap_x_max = min(bb_x_max, col['x_end'])
                overlap_y_min = max(bb_y_min, col['y_start'])
                overlap_y_max = min(bb_y_max, col['y_end'])

                overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)

                overlapping_columns.append({
                    'col': col,
                    'overlap_area': overlap_area,
                    'x_min': overlap_x_min,
                    'x_max': overlap_x_max,
                    'y_min': overlap_y_min,
                    'y_max': overlap_y_max
                })

        if not overlapping_columns:
            # Box doesn't overlap with any column - it's OUTSIDE
            outside_results.append((bbox, text, conf))
            print(f"🔴 OUTSIDE: '{text}' at ({int(bb_x_min)}, {int(bb_y_min)})")
            continue

        if len(overlapping_columns) == 1:
            # Box is entirely within one column, clip to column bounds
            overlap = overlapping_columns[0]
            clipped_bbox = [
                [overlap['x_min'], overlap['y_min']],
                [overlap['x_max'], overlap['y_min']],
                [overlap['x_max'], overlap['y_max']],
                [overlap['x_min'], overlap['y_max']]
            ]
            clipped_results.append((clipped_bbox, text, conf, overlap['col']['id']))

            # Check if clipping occurred
            if (overlap['x_min'] > bb_x_min or overlap['x_max'] < bb_x_max or
                overlap['y_min'] > bb_y_min or overlap['y_max'] < bb_y_max):
                clipped_count += 1
                print(f"✂️  CLIPPED: '{text}' to column {overlap['col']['id']}")
        else:
            # Box spans multiple columns, split it
            split_count += 1
            print(f"✂️  SPLIT: '{text}' across {len(overlapping_columns)} columns")

            # Sort by overlap area (largest first) to assign text to primary column
            overlapping_columns.sort(key=lambda x: x['overlap_area'], reverse=True)

            for idx, overlap in enumerate(overlapping_columns):
                clipped_bbox = [
                    [overlap['x_min'], overlap['y_min']],
                    [overlap['x_max'], overlap['y_min']],
                    [overlap['x_max'], overlap['y_max']],
                    [overlap['x_min'], overlap['y_max']]
                ]

                # Assign full text to primary column (largest overlap), empty to others
                split_text = text if idx == 0 else ""
                clipped_results.append((clipped_bbox, split_text, conf, overlap['col']['id']))
                print(f"     → Column {overlap['col']['id']}: " +
                      f"({int(overlap['x_min'])}, {int(overlap['y_min'])}) to " +
                      f"({int(overlap['x_max'])}, {int(overlap['y_max'])})" +
                      (f" [text: '{split_text}']" if split_text else " [no text]"))

    print(f"\n✨ Overflow Processing: {split_count} boxes split, {clipped_count} boxes clipped")
    print(f"   Total OCR boxes inside columns: {len(clipped_results)}")
    print(f"   Total OCR boxes OUTSIDE columns: {len(outside_results)}")

    return clipped_results, outside_results


def draw_all_columns(img: np.ndarray, columns: List[Dict], box_width: int = 3) -> np.ndarray:
    """
    Draw all detected columns with different colored bounding boxes.

    Args:
        img: Input image
        columns: List of column dictionaries
        box_width: Width of bounding box lines

    Returns:
        Image with all columns drawn
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]

    result = img.copy()

    for idx, col in enumerate(columns):
        color = colors[idx % len(colors)]
        cv2.rectangle(result,
                     (col['x_start'], col['y_start']),
                     (col['x_end'], col['y_end']),
                     color, box_width)

        # Add column ID label
        label = f"Col {col['id']}"
        label_pos = (col['x_start'] + 5, col['y_start'] + 25)
        cv2.putText(result, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, color, 2)

    print(f"\n🎨 Drew {len(columns)} columns with colored bounding boxes")
    return result


def draw_clipped_ocr_boxes(img: np.ndarray, clipped_ocr: List[Tuple],
                           columns: List[Dict], box_width: int = 2) -> np.ndarray:
    """
    Draw all clipped OCR bounding boxes with column-matched colors.

    Args:
        img: Input image
        clipped_ocr: List of (bbox, text, confidence, column_id) tuples
        columns: List of column dictionaries
        box_width: Width of bounding box lines

    Returns:
        Image with OCR boxes drawn
    """
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 128, 0),  # Orange
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]

    result = img.copy()

    for (bbox, text, conf, col_id) in clipped_ocr:
        color = colors[col_id % len(colors)]
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

        # Optionally draw text label
        if text:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            text_pos = (int(min(x_coords)) + 2, int(min(y_coords)) + 12)
            cv2.putText(result, text[:20], text_pos, cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, color, 1)

    print(f"🎨 Drew {len(clipped_ocr)} clipped OCR boxes")
    return result


def draw_outside_ocr_boxes(img: np.ndarray, outside_ocr: List[Tuple],
                          box_width: int = 3, color: Tuple = (0, 0, 255)) -> np.ndarray:
    """
    Draw OCR bounding boxes that are outside all columns.

    Args:
        img: Input image
        outside_ocr: List of (bbox, text, confidence) tuples for outside text
        box_width: Width of bounding box lines
        color: Color for outside boxes (default: red)

    Returns:
        Image with outside OCR boxes drawn
    """
    result = img.copy()

    for (bbox, text, conf) in outside_ocr:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=box_width)

    print(f"🎨 Drew {len(outside_ocr)} outside OCR boxes in red")
    return result

def mask_column_by_text(filename: str,
                        search_terms: List[str],
                        outside_search_terms: List[str] = [],
                        horizontal_kernel_size: int = 50,
                        vertical_kernel_size: int = 50,
                        box_width: int = 5,
                        language: List[str] = ['en'],
                        gpu: bool = True,
                        merge_distance: float = 20,
                        overlay_alpha: float = 0.6):
    """
    Find and highlight columns containing specific text in a table image.
    Also detects and highlights text outside column boundaries.

    Args:
        filename: Path to the input image
        search_terms: List of terms to search for in column headers (case-insensitive, exact match)
        outside_search_terms: List of terms to search for in text outside columns (case-insensitive, exact match)
        horizontal_kernel_size: Size of horizontal line detection kernel
        vertical_kernel_size: Size of vertical line detection kernel
        box_width: Width of bounding box lines
        language: List of languages for OCR
        gpu: Whether to use GPU for OCR
        merge_distance: Distance threshold for merging OCR boxes
        overlay_alpha: Transparency of overlay boxes (0=transparent, 1=opaque)

    Returns:
        Dictionary with visualization images:
        {
            'all_columns': image with all columns,
            'clipped_ocr': image with clipped OCR boxes,
            'outside_ocr': image with outside OCR boxes highlighted,
            'final_result': image with matched columns highlighted
        }
    """

    # === STEP 1: Detect table structure ===
    img = cv2.imread(filename)
    if img is None:
        raise ValueError(f"Could not read image from {filename}")

    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Detect lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Extract and merge line segments
    def extract_segments(binary_img, is_horizontal):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if is_horizontal:
                segments.append({'start': x, 'end': x + w, 'position': y + h // 2})
            else:
                segments.append({'start': y, 'end': y + h, 'position': x + w // 2})
        return segments

    def merge_segments(segments, pos_tol, gap_tol, img_size):
        if not segments:
            return []
        segments = sorted(segments, key=lambda s: (s['position'], s['start']))
        merged = []
        for seg in segments:
            merged_flag = False
            for existing in merged:
                pos_diff = abs(seg['position'] - existing['position'])
                if pos_diff <= img_size * pos_tol:
                    gap = max(seg['start'], existing['start']) - min(seg['end'], existing['end'])
                    if gap <= img_size * gap_tol:
                        existing['start'] = min(existing['start'], seg['start'])
                        existing['end'] = max(existing['end'], seg['end'])
                        merged_flag = True
                        break
            if not merged_flag:
                merged.append(seg.copy())
        return merged

    h_segs = extract_segments(h_lines, True)
    v_segs = extract_segments(v_lines, False)
    h_merged = merge_segments(h_segs, 0.02, 0.10, height)
    v_merged = merge_segments(v_segs, 0.02, 0.10, width)

    # Find intersections
    intersections = []
    for h in h_merged:
        for v in v_merged:
            if h['start'] <= v['position'] <= h['end'] and v['start'] <= h['position'] <= v['end']:
                intersections.append((v['position'], h['position']))

    # Extract columns
    if not intersections:
        print("No table structure detected")
        return {'all_columns': img, 'clipped_ocr': img, 'outside_ocr': img, 'final_result': img}

    x_vals = sorted(set([pt[0] for pt in intersections]))
    y_vals = [pt[1] for pt in intersections]
    y_min, y_max = min(y_vals), max(y_vals)

    # Merge similar x values
    merged_x = []
    for x in x_vals:
        if not merged_x or abs(x - merged_x[-1]) > width * 0.02:
            merged_x.append(x)
        else:
            merged_x[-1] = (merged_x[-1] + x) // 2

    columns = []
    for i in range(len(merged_x) - 1):
        columns.append({
            'id': i,
            'x_start': merged_x[i],
            'x_end': merged_x[i + 1],
            'y_start': y_min,
            'y_end': y_max
        })

    print(f"Detected {len(columns)} columns")

    # Draw all columns
    all_columns_img = draw_all_columns(img, columns, box_width=3)

    # === STEP 2: OCR and preprocessing ===
    print(f"\nRunning OCR...")
    reader = easyocr.Reader(language, gpu=gpu)
    ocr_results = reader.readtext(filename)
    print(f"Initial OCR detected {len(ocr_results)} text boxes")

    # Merge OCR results
    print(f"\nMerging OCR results...")
    merged_ocr = merge_ocr_results(ocr_results, merge_distance)

    # Clip and split OCR boxes, also get outside boxes
    print(f"\nClipping and splitting OCR boxes to column boundaries...")
    clipped_ocr, outside_ocr = clip_and_split_ocr_boxes(merged_ocr, columns)

    # Draw clipped OCR boxes
    clipped_ocr_img = draw_clipped_ocr_boxes(all_columns_img.copy(), clipped_ocr,
                                             columns, box_width=2)

    # Draw outside OCR boxes
    outside_ocr_img = draw_outside_ocr_boxes(all_columns_img.copy(), outside_ocr,
                                            box_width=3, color=(0, 0, 255))

    # === STEP 3: Separate headers from data ===
    # Store OCR results per column
    ocr_by_column = {i: [] for i in range(len(columns))}

    for (bbox, text, conf, col_id) in clipped_ocr:
        ocr_by_column[col_id].append((bbox, text, conf))

    headers_by_column = {}
    data_by_column = {}

    for col_id, ocr_boxes in ocr_by_column.items():
        if not ocr_boxes:
            continue

        # Sort by y position to find topmost box (header)
        sorted_boxes = sorted(ocr_boxes, key=lambda x: min(p[1] for p in x[0]))

        # First box is the header
        headers_by_column[col_id] = sorted_boxes[0]

        # Rest are data
        data_by_column[col_id] = sorted_boxes[1:] if len(sorted_boxes) > 1 else []

        if sorted_boxes:
            header_text = sorted_boxes[0][1]
            print(f"Column {col_id} header: '{header_text}' with {len(sorted_boxes)-1} data rows")

    # === STEP 4: Search for EXACT matches in HEADERS only ===
    search_normalized = [term.strip().lower() for term in search_terms]
    matched_cols = []
    matched_ids = set()

    # Track which terms matched which columns
    matches_by_term = {term: [] for term in search_terms}

    # Search in HEADERS with EXACT match
    print("\n🔍 Searching for EXACT matches in column headers...")
    for col_id, (bbox, text, conf) in headers_by_column.items():
        text_normalized = text.strip().lower()

        # Check against all search terms with EXACT match
        for original_term, normalized_term in zip(search_terms, search_normalized):
            if normalized_term == text_normalized:  # EXACT MATCH
                # Find the column object
                col = next((c for c in columns if c['id'] == col_id), None)
                if col and col['id'] not in matched_ids:
                    matched_cols.append(col)
                    matched_ids.add(col['id'])
                matches_by_term[original_term].append(('column', col_id, text))
                print(f"✅ EXACT MATCH: Found '{text}' (matching '{original_term}') in Column {col_id} header")

    # === STEP 5: Search for EXACT matches in OUTSIDE text AND HEADERS ===
    outside_search_normalized = [term.strip().lower() for term in outside_search_terms]
    matched_outside = []

    # Track which outside terms matched
    outside_matches_by_term = {term: [] for term in outside_search_terms}

    print("\n🔍 Searching for matches in outside text AND headers...")

    # Search in OUTSIDE text boxes
    for (bbox, text, conf) in outside_ocr:
        text_normalized = text.strip().lower()
        # Also create a version without special characters for flexible matching
        text_alphanum = ''.join(c for c in text_normalized if c.isalnum())

        for original_term, normalized_term in zip(outside_search_terms, outside_search_normalized):
            term_alphanum = ''.join(c for c in normalized_term if c.isalnum())
            
            if normalized_term == text_normalized or term_alphanum == text_alphanum or normalized_term in text_normalized:
                matched_outside.append((bbox, text, conf))
                outside_matches_by_term[original_term].append(('outside', None, text))
                print(f"✅ MATCH: Found '{text}' (matching '{original_term}') OUTSIDE columns")
                break

    # Log if search terms appear in headers (for debugging) but DON'T mask them
    for col_id, (bbox, text, conf) in headers_by_column.items():
        text_normalized = text.strip().lower()
        text_alphanum = ''.join(c for c in text_normalized if c.isalnum())

        for original_term, normalized_term in zip(outside_search_terms, outside_search_normalized):
            term_alphanum = ''.join(c for c in normalized_term if c.isalnum())
            
            if normalized_term == text_normalized or term_alphanum == text_alphanum or normalized_term in text_normalized:
                # DON'T add to matched_outside - just log it
                outside_matches_by_term[original_term].append(('header_found_not_masked', col_id, text))
                print(f"ℹ️  INFO: Found '{text}' (matching '{original_term}') in Column {col_id} HEADER (not masking)")
                break

    # === STEP 6: Draw visualization - Cover DATA ONLY with white and *** ===
    result = img.copy()

    # Cover data boxes ONLY in matched columns with white and ***
    # Headers are NOT masked
    for col_id in matched_ids:
        data_boxes = data_by_column.get(col_id, [])
        for (bbox, text, conf) in data_boxes:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            # Draw filled white rectangle to cover the bounding box
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

            # Draw *** left-aligned with some padding
            padding = 5
            text_y = y_min + (y_max - y_min + 12) // 2  # Vertically centered
            cv2.putText(result, "***", (x_min + padding, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if matched_ids:
        print(f"\n🔒 Covered DATA boxes in {len(matched_ids)} matched column(s) with white and ***")
        print(f"   Headers were NOT masked")

    # Cover matched outside boxes and extend 300px to the right
    if matched_outside:
        outside_color = (255, 255, 255)  # White for covering
        for (bbox, text, conf) in matched_outside:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            # Extend 300px to the right from the end of the box (but don't exceed image width)
            extended_x_max = min(x_max + 300, width)

            # Draw filled white rectangle covering the bounding box and extension
            cv2.rectangle(result, (x_max, y_min), (extended_x_max, y_max),
                        outside_color, -1)

            # Draw *** left-aligned in the original box area with padding
            padding = 5
            text_y = y_min + (y_max - y_min + 12) // 2  # Vertically centered
            cv2.putText(result, "***", (x_max + padding + 50, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            print(f"🔒 Covered outside box '{text}' and extended 300px right")

        print(f"✅ Processed {len(matched_outside)} matched outside box(es)")

    # Print summary
    print(f"\n{'='*60}")
    print("SEARCH SUMMARY")
    print(f"{'='*60}")

    # Column header search results
    if search_terms:
        print("\n📋 COLUMN HEADER SEARCH:")
        for term, matches in matches_by_term.items():
            if matches:
                col_matches = [m[1] for m in matches if m[0] == 'column']
                match_str = f"  '{term}':"
                if col_matches:
                    match_str += f" ✅ Found in column header(s) {sorted(set(col_matches))}"
                print(match_str)
            else:
                print(f"  '{term}': ❌ Not found (exact match required)")

    # Outside text search results
    if outside_search_terms:
        print("\n🔍 OUTSIDE TEXT SEARCH:")
        for term, matches in outside_matches_by_term.items():
            if matches:
                print(f"  '{term}': ✅ Found OUTSIDE columns ({len(matches)} match(es))")
            else:
                print(f"  '{term}': ❌ Not found (exact match required)")

    if not matched_cols and not matched_outside:
        print(f"\n❌ No EXACT matches found for any search terms")
        print(f"   Note: Search is case-insensitive but requires exact text match after lowering/stripping")

    return {
        'all_columns': all_columns_img,
        'clipped_ocr': clipped_ocr_img,
        'outside_ocr': outside_ocr_img,
        'final_result': result
    }


# Example usage
if __name__ == "__main__":
    # Input image path
    input_image = 'src/images/ss-2.jpeg'
    
    results = mask_column_by_text(
        filename=input_image,
        search_terms=['line number', 'description',],
        outside_search_terms=['order number', 'sold to', 'ship to',],
        box_width=5,
        merge_distance=5,
        overlay_alpha=0.6
    )

    # Get the directory and base name of the input image
    import os
    input_dir = os.path.dirname(input_image)
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    
    # Save results in the same directory as input image
    output_paths = {
        # 'all_columns': os.path.join(input_dir, f'{base_name}_columns.jpg'),
        # 'clipped_ocr': os.path.join(input_dir, f'{base_name}_clipped_ocr.jpg'),
        # 'outside_ocr': os.path.join(input_dir, f'{base_name}_outside_ocr.jpg'),
        'final_result': os.path.join(input_dir, f'{base_name}_final_result.jpg')
    }
    
    # Save all images
    # cv2.imwrite(output_paths['all_columns'], results['all_columns'])
    # cv2.imwrite(output_paths['clipped_ocr'], results['clipped_ocr'])
    # cv2.imwrite(output_paths['outside_ocr'], results['outside_ocr'])
    cv2.imwrite(output_paths['final_result'], results['final_result'])
    
    # print("\n" + "="*60)
    # print("SAVED OUTPUT IMAGES")
    # print("="*60)
    # for name, path in output_paths.items():
    #     print(f"{name}: {path}")
    
    # # Display results (Windows-compatible)
    # print("\n" + "="*60)
    # print("DISPLAYING RESULTS (Press any key to continue)")
    # print("="*60)
    
    # cv2.imshow('1. All Detected Columns', results['all_columns'])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('2. Clipped OCR Boxes', results['clipped_ocr'])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('3. Outside OCR Boxes', results['outside_ocr'])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.imshow('4. Final Result', results['final_result'])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # print("\n✅ Processing complete!")