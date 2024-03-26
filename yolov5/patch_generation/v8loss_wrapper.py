import torch
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou, probiou
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from easydict import EasyDict as edict


class v8DetectionLossWrapper(v8DetectionLoss):
    def __init__(self, model, n_classes):
        """Initializes v8DetectionLoss with a yolov5 PyTorch model and sets attributes respectively."""
        device = next(model.parameters()).device  # get model device
        h = edict({'box': 7.5, 'cls': 0.5, 'dfl': 1.5})  # see github.com/ultralytics/ultralytics/ -> yolo/cfg/default.yaml

        self.m = model  # Detect() module
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = model.stride  # model strides
        self.nc = n_classes  # number of classes
        self.reg_max = 16  # see https://github.com/ultralytics/ultralytics -> ultralytics/nn/modules/head.py
        self.no = self.nc + self.reg_max * 4
        self.device = device

        self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)


