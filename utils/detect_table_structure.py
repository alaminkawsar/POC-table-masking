import cv2
import numpy as np
from utils.drawings import draw_all_columns
from typing import List, Dict

class TableDetector:
    def __init__(self, horizontal_kernel_size=50, vertical_kernel_size=50):
        self.horizontal_kernel_size = horizontal_kernel_size
        self.vertical_kernel_size = vertical_kernel_size
        
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
    
    
    def detect(self, filename: str):
        # === STEP 1: Detect table structure ===
        img = cv2.imread(filename)
        if img is None:
            raise ValueError(f"Could not read image from {filename}")

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

        # Detect lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.horizontal_kernel_size, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.vertical_kernel_size))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        h_segs = self.extract_segments(h_lines, True)
        v_segs = self.extract_segments(v_lines, False)
        h_merged = self.merge_segments(h_segs, 0.02, 0.10, height)
        v_merged = self.merge_segments(v_segs, 0.02, 0.10, width)
        
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
    
    def _make_table_structure(self):
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
                
    def _search_term(self, search_terms: List[str]):
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