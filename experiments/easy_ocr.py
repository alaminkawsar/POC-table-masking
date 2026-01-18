# QUICK AND SIMPLE OCR WITH BOUNDING BOXES
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np

def quick_ocr(image_path, languages=['en']):
    """Simple OCR with visualization"""
    
    # Initialize reader
    reader = easyocr.Reader(languages)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    
    # Perform OCR
    results = reader.readtext(img)
    
    # Create a copy for annotation
    annotated = img.copy()
    
    # Draw bounding boxes and text
    colors = {
        'box': (0, 255, 0),      # Green
        'text': (0, 0, 255),     # Red
        'conf': (255, 165, 0)    # Orange
    }
    
    for (bbox, text, confidence) in results:
        # Convert coordinates to integers
        pts = [[int(x), int(y)] for x, y in bbox]
        
        # Draw bounding box
        cv2.polylines(annotated, [np.array(pts)], True, colors['box'], 2)
        
        # Prepare text label
        label = f"{text} ({confidence:.0%})"
        
        # Position for text (above bbox)
        text_x = pts[0][0]
        text_y = pts[0][1] - 10
        if text_y < 0:
            text_y = pts[3][1] + 30
        
        # Draw text
        cv2.putText(annotated, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Annotated
    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'OCR Results - {len(results)} regions found')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nFound {len(results)} text regions:")
    for i, (bbox, text, conf) in enumerate(results):
        print(f"{i+1}. '{text}' (confidence: {conf:.1%})")
    
    return results

# Usage
results = quick_ocr('/home/kawsar/Desktop/Image-Masking/POC-table-masking/data/images/21.JPG')