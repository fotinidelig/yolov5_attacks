import torch
import numpy
import torchvision
from torchvision.ops import nms
import torchvision.ops.boxes as box_ops
from typing import List, Union, Tuple
from yolov5.patch_generation.loss import CopmuteLoss
from yolov5.patch_generation.utils import NPSLoss


def optimization_step(batch, patch, targets, model):
    """
    batch: torch.Tensor image batch
    patch: torch.Tensor with requires_grad_(True)
    targets: torch.Tensor of shape (N, 6) N number of detected objects in entire batch
    model: YOLO model
    """
    n_samples = len(batch)
    preds = model(batch)


