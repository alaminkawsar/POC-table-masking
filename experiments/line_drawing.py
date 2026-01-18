import os
import cv2
import numpy as np
import math

IMGE_FOLDER = '/home/kawsar/Desktop/Image-Masking/POC-table-masking/data/images'
OUTPUT_FOLDER = '/home/kawsar/Desktop/Image-Masking/POC-table-masking/data/temp'

def line_detection(image_path):
    # 1. Load the image in grayscale
    # Replace 'image.jpg' with your image file path
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Apply Canny edge detection
    # The thresholds (50, 150) can be adjusted
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 3. Use the Probabilistic Hough Line Transform (HoughLinesP)
    # Adjust parameters (threshold, minLineLength, maxLineGap) for best results
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,  # Minimum number of votes needed to be considered a line
        minLineLength=50, # Minimum length of a line
        maxLineGap=10   # Maximum gap between line segments to be treated as a single line
    )

    # 4. Draw the detected lines on the original color image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Draw the line in green with a thickness of 2
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def main():
    # traverse each image in the folder
    for filename in os.listdir(IMGE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(IMGE_FOLDER, filename)
            result_img = line_detection(image_path)
            output_path = os.path.join(OUTPUT_FOLDER, f"lines_{filename}")
            cv2.imwrite(output_path, result_img)
            print(f"Processed {filename}, saved to {output_path}")
            
if __name__ == "__main__":
    main()