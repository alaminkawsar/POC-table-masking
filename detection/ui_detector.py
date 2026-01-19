from ultralytics import YOLO

class UIElementDetector:
  def __init__(self, model_path: str = "assets/best.pt"):
    self.model = YOLO(model_path)

  def detect_ui_elements(self, image_path: str):
    if self.model is None:
      raise RuntimeError("Model not loaded. Call load() first.")
    result = self.model.predict(image_path)

    extracted_data = []
    CLASS_NAMES = {
        0: "table_column",
        1: "text_field",
        2: "text_info"
    }

    if result[0].boxes is not None:
        for box in result[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            label = CLASS_NAMES.get(cls_id, f"unknown_class_{cls_id}")
            extracted_data.append({
                "box": [x1, y1, x2, y2],
                "label": label
            })
    return extracted_data
