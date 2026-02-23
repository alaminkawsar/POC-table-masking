import os
from main import main

IMAGE_FOLDER = ""

for image_name in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    main(
        headers_text="",
        image_path=image_path,
        model_path="assets/best.pt"
    )