import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from skimage.draw import rectangle_perimeter
import numpy as np
import cv2
from time import time

model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
image = Image.open("hotdog.webp")

# Check for cats and remote controls
text = "a person. a car. a truck. a bike."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)

start = time()
with torch.no_grad():
    outputs = model(**inputs)

print(f"Inference took {time() - start} sec")

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.2,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
)
print(results)

boxes: torch.Tensor = results[0]["boxes"]

boxes = boxes.numpy(force=True)

labels = results[0]["labels"]

colors = {
    "a truck": (255, 0, 0),
    "a person": (0, 255, 0),
    "a car": (0, 0, 255),
    "a car a": (0, 0, 255),
    "a bike": (255, 0, 255),
    "a hot dog": (255, 0, 255),
}
image = np.asarray(image).copy()
i = 0
for box in boxes:
    box = box.astype(int)
    label = labels[i]
    print(box, label)
    start = box[0:2]
    end = box[2:]
    # rr, cc = rectangle_perimeter(start, end=end, shape=image.shape)

    try:
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), colors[label], 32
        )
    except KeyError as e:
        print(e)
        # continue
    # image[rr, cc] = [255, 0, 0]
    i += 1

Image.fromarray(image).save("results.jpg")
