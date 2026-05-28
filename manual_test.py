"""Standalone end-to-end test of the RF-DETR Scope pipeline.

Downloads a known sample image, runs it through the pipeline on the GPU,
prints detections, and saves an annotated output frame.
"""

import io

import numpy as np
import requests
import torch
from PIL import Image
from rfdetr.util.coco_classes import COCO_CLASSES

from scope_rfdetr_detection.pipelines.pipeline import RFDetrDetectionPipeline

URL = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

print("torch.cuda.is_available:", torch.cuda.is_available())
print("downloading test image...")
img = Image.open(io.BytesIO(requests.get(URL, timeout=60).content)).convert("RGB")
print("image size:", img.size)

print("loading pipeline (nano)...")
pipe = RFDetrDetectionPipeline(model_variant="nano")
print("pipeline device:", pipe.device)

# Direct detection report
dets = pipe.model.predict(img, threshold=0.5)
print("num detections:", len(dets))
for i in range(len(dets)):
    cid = int(dets.class_id[i])
    conf = float(dets.confidence[i])
    print(f"  - {COCO_CLASSES.get(cid, cid)}  {conf:.2f}")

# Full pipeline path (frame tensor (1,H,W,3) uint8 -> annotated (T,H,W,3) float[0,1])
frame = torch.from_numpy(np.array(img)).unsqueeze(0)
out = pipe(video=[frame], confidence_threshold=0.5, show_labels=True, show_confidence=True)
v = out["video"]
print("output tensor:", tuple(v.shape), v.dtype, "range", float(v.min()), float(v.max()))

annotated = (v[0].numpy() * 255).astype(np.uint8)
Image.fromarray(annotated, "RGB").save("rfdetr_test_out.jpg")
print("saved rfdetr_test_out.jpg")
