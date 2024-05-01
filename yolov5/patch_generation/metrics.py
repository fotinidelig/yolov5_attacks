import os
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from yolov5.patch_generation import CONFIG_FILE
import configparser
from yolov5.utils.general import non_max_suppression
from yolov5.val import process_batch
from yolov5.utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from yolov5.utils.general import (
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
)

FILE_PATH = Path(os.path.abspath(__file__)).parent

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

NC = config.get("data", "num_classes")
CONF_THRES, iou_thres=IOU_THRES, IOU_THRES = .6, .6



def get_eval_metrics(preds, targets, im_shape, device='cuda'):
    targets[:, 2:] *= torch.tensor((im_shape[0], im_shape[1], im_shape[0], im_shape[1]), device=device)  # from normalized xy values to pixels
    preds = non_max_suppression(preds, CONF_THRES, iou_thres=IOU_THRES, IOU_THRES)


    stats = []
    iouv = torch.linspace(IOU_THRES, 0.95, 510, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    confusion_matrix = ConfusionMatrix(nc=NC)

    for si, pred in enumerate(preds): # for each image
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
        correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init matrix with number of correct predictions per iou threshold

        if npr == 0:
            if nl: # no predictions made, but image contains ground truth labels
                stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            continue

        # Predictions
        predn = pred.clone()
        scale_boxes(im_shape, predn[:, :4], shape, shapes[si][1])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            scale_boxes(im_shape, tbox, shape, shapes[si][1])  # native-space labels
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            confusion_matrix.process_batch(predn, labelsn)
        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)


def get_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, im_shape: Tuple[int, int], grid_shape: Tuple[int, int], device: str='cuda'):
    """_summary_
    Gets confusion_matrix for 
    Args:
        preds (torch.Tensor): model predictions for a single resolution, shape (bsize, im_shape, im_shape, 5+num_classes) where 5 comes from (x,y,w,h,obj_conf)
        targets (torch.Tensor): ground truths, (img_id, cls_id, x, y, w, h) (normalized coodrinates)
        im_shape (Tuple[int, int]): original input image shape
        grid_shape (Tuple[int, int]): grid shape (80 or 40 or 20 resolution)
        device (str, optional): Defaults to 'cuda'
    """
    confusion_matrix = ConfusionMatrix(nc=NC, conf=CONF_THRES, iou_thres=IOU_THRES)



