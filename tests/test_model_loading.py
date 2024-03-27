from yolov5.models.yolo import Model
from yolov5.models.common import DetectMultiBackend
from pathlib import Path
import os
import torch

FILE_PATH = Path(os.path.abspath(__file__)).parent

# Load model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# model = DetectMultiBackend(weights="../../yolov5s.pt", dnn=False,
#                            data="../data/coco.yaml", fp16=False)

model = Model(FILE_PATH / ".." / "models" / "yolov5s.yaml", ch=3, nc=80, anchors=None)
assert model is not None