import os
import torch
import torchvision
from tqdm import tqdm
from pathlib import Path
from torchvision.ops import nms
import torchvision.ops.boxes as box_ops
from typing import List, Union, Tuple
from yolov5.patch_generation.loss import ComputeLoss, extract_patch_grads, NPSLoss
from yolov5.patch_generation.utils import YOLODataset, apply_patch

FILE_PATH = Path(os.path.abspath(__file__)).parent

def optimization(model: torch.nn.Module, hyp: dict, loader: torch.utils.data.DataLoader,
                 patch: torch.Tensor, optimizer: torch.optim.Optimizer,
                 device: str="cuda"):
    """
    batch: torch.Tensor image batch
    patch: torch.Tensor with requires_grad_(True)
    targets: torch.Tensor of shape (N, 6) N number of detected objects in entire batch
    model: YOLO model
    """

    total_batches = len(loader)
    # Define loss functions
    # TODO: add more losses beside detection & non-printability loss
    detection_loss_fn = ComputeLoss(hyp, device=device)

    nps_loss_fn = NPSLoss(FILE_PATH / "30_rgb_triplets.csv", patch.shape[1:], device=device)

    # TODO: fine-tune the weighting of the different losses
    w_det = 1.
    w_nps = 1.
    with tqdm(loader, total=total_batches, desc='Input Batches') as pbar:
        for b, batch in enumerate(pbar):
            inputs, labels = batch
            targets = YOLODataset.labels_to_targets(labels).to(device)
            inputs = inputs.to(device)

            # Patch application
            patch_application = [apply_patch(inputs[i], patch.clone()) for i in range(len(inputs))]
            patched_inputs = torch.stack([a[0] for a in patch_application])
            patched_inputs.requires_grad_(True)
            patch_positions = torch.stack([a[1] for a in patch_application])

            preds = model(patched_inputs)
            # Loss calculation
            det_loss, box_obj_cls_loss = detection_loss_fn(preds, targets)
            det_loss = det_loss/len(patched_inputs) # mean detection loss
            det_loss.backward()

            nps_loss = nps_loss_fn(patch)
            nps_loss.backward()

            # Grad calculation
            patch_det_grads = torch.stack([extract_patch_grads(img_grads, pos, patch.shape[1:])
                               for img_grads, pos in zip(patched_inputs.grad, patch_positions)])
            patch_det_grad = torch.mean(patch_det_grads, dim=0)
            patch.grad = w_nps * patch.grad + w_det * patch_det_grad
            # Optimization step
            optimizer.step()
            optimizer.zero_grad()

            # Monitoring
            pbar.set_postfix(det_loss=det_loss.item(), nps_loss=nps_loss.item())
            # pbar.update(1)
    return patch, det_loss.item(), nps_loss.item()
