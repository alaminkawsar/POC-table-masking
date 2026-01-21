import easyocr
from .base_ocr import BaseOCR

class EasyOCR(BaseOCR):
    def __init__(self, lang_list=None, gpu=False):
        if lang_list is None:
            lang_list = ['en']
        self.reader = easyocr.Reader(lang_list, gpu=gpu)

    def read_text(self, image):
        results = self.reader.readtext(image, detail=0)
        return ' '.join(results)

    def read_text_with_boxes(self, image):
        results = self.reader.readtext(image, detail=1)
        output = []
        for bbox, text, conf in results:
            output.append((text, bbox, conf))
        return output

    def detect_text_regions(self, image):
        results = self.reader.readtext(image, detail=1)
        boxes = []
        for bbox, text, conf in results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y
            boxes.append((x, y, w, h))
        return boxes