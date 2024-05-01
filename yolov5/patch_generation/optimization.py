import os
import torch
import mlflow
import torchvision
from tqdm import tqdm
from pathlib import Path
import configparser
from torchvision.ops import nms
from yolov5.patch_generation import CONFIG_FILE, check_dir
from yolov5.patch_generation.loss import ComputeLoss, extract_patch_grads, NPSLoss
from yolov5.patch_generation.utils import (
    YOLODataset,
    batch_apply_patch,
    PatchTransform,
    register_numerical_hooks,
    apply_patch_and_pad_batch,
)

FILE_PATH = Path(os.path.abspath(__file__)).parent

config = configparser.ConfigParser()
config.read(CONFIG_FILE)
VAL_STEP = int(config.get("train", "val_step"))
RUN_DIR = config.get("train", "run_dir")
MODEL_IN_SZ = config.get("model", "model_in_sz")
check_dir(RUN_DIR)
check_dir(f"{RUN_DIR}/validation/")


def optimization(
    model: torch.nn.Module,
    hyp: dict,
    dataset: YOLODataset,
    loader: torch.utils.data.DataLoader,
    valloader: torch.utils.data.DataLoader,
    patch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    device: str = "cuda",
):
    """
    batch: torch.Tensor image batch
    patch: torch.Tensor with requires_grad_(True)
    targets: torch.Tensor of shape (N, 6) N number of detected objects in entire batch
    model: YOLO model
    """

    total_batches = len(loader)

    # Patch transformer
    patch_transform = PatchTransform(patch, device=device)

    # Define loss functions
    # TODO: add more losses beside detection & non-printability loss
    detection_loss_fn = ComputeLoss(hyp, device=device)

    nps_loss_fn = NPSLoss(
        FILE_PATH / "30_rgb_triplets.csv", patch.shape[1:], device=device
    )

    # TODO: fine-tune the weighting of the different losses
    w_det = 10
    w_nps = 1

    ebar = tqdm(range(epochs), desc="Epochs")
    for epoch in ebar:
        pbar = tqdm(loader, total=total_batches, desc="Input Batches")
        metrics = {"det_loss": 0, "nps_loss": 0, "val_det_loss": 0}

        for b, batch in enumerate(pbar):
            pbar.set_description(f"Input Batches ep. {epoch+1}/{ebar.total}")
            inputs, labels = batch

            # Patch transformation
            patches = patch_transform(len(inputs))

            # Apply patch, afterwards pad images and labels to model input size
            patched_inputs, targets, patch_positions = apply_patch_and_pad_batch(
                dataset, inputs, patches, labels, device
            )
            patched_inputs.requires_grad_(True)
            register_numerical_hooks(
                patched_inputs
            )  # solves errors due to NaN and inf values

            # Loss calculation
            nps_loss = nps_loss_fn(patch)
            nps_loss.backward()

            preds = model(patched_inputs)
            det_loss, box_obj_cls_loss = detection_loss_fn(preds, targets)
            det_loss = det_loss / len(patched_inputs)  # mean detection loss
            det_loss.backward()

            # Grad calculation
            patch_det_grads = torch.stack(
                [
                    extract_patch_grads(img_grads, pos, patch.shape[1:])
                    for img_grads, pos in zip(patched_inputs.grad, patch_positions)
                ]
            )
            patch_det_grad = torch.mean(patch_det_grads, dim=0)
            patch.grad = w_nps * patch.grad - w_det * patch_det_grad

            # Optimization step
            if torch.isnan(patch.grad.sum()):
                optimizer.zero_grad()
                pbar.set_postfix({"batch_error": "NaN grads"})
                continue  # avoid updating patch with NaN gradients

            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0.000001, 1.0)

            optimizer.zero_grad()

            # Monitoring
            metrics["det_loss"] += det_loss.detach().item()
            metrics["nps_loss"] += nps_loss.detach().item()
            if b < total_batches:
                # plot mean loss intstead at the final epoch
                pbar.set_postfix(
                    det_loss=det_loss.detach().item(), nps_loss=nps_loss.detach().item()
                )

            del det_loss, nps_loss, inputs, patched_inputs, preds

        metrics["det_loss"] /= b  # mean loss of b batches
        metrics["nps_loss"] /= b

        pbar.set_postfix(det_loss=metrics["det_loss"], nps_loss=metrics["nps_loss"])
        ebar.set_postfix(det_loss=metrics["det_loss"], nps_loss=metrics["nps_loss"])

        mlflow.log_metric("det_loss", metrics["det_loss"], epoch)
        mlflow.log_metric("nps_loss", metrics["nps_loss"], epoch)

        if epoch % VAL_STEP != 0 and epoch - 1 != epochs:  # no validation
            continue

        # Enter validation
        torch.save(patch, f"{RUN_DIR}/validation/patch_{epoch}.pt")
        torchvision.utils.save_image(patch, f"{RUN_DIR}/validation/patch_{epoch}.png")
        # save_patch(patch.detach().clone().cpu(), f"{RUN_DIR}/patch_{epoch}")
        pbar = tqdm(valloader, total=len(valloader), desc="Validation Batches")

        for b, batch in enumerate(pbar):
            inputs, labels = batch

            # Patch transformation
            patches = patch_transform(len(inputs))

            # Apply patch, afterwards pad images and labels to model input size
            patched_inputs, targets, patch_positions = apply_patch_and_pad_batch(
                dataset, inputs, patches, labels, device
            )

            # Loss calculation
            nps_loss = nps_loss_fn(patch)

            preds = model(patched_inputs)
            det_loss, _ = detection_loss_fn(preds, targets)

            # Monitoring

            metrics["val_det_loss"] += det_loss.detach().item()

            if (
                epoch == 0 and b == 0
            ):  # if first epoch and first val. batch, save images as examples
                torchvision.utils.save_image(
                    [im for im in patched_inputs],
                    f"{RUN_DIR}/validation/pathed_ims.png",
                    nrow=5,
                )
            pbar.set_postfix(det_loss=det_loss.detach().item())

            del det_loss, nps_loss, inputs, patched_inputs, preds

        metrics["val_det_loss"] /= b  # mean loss of b batches

        mlflow.log_metric("val_det_loss", metrics["val_det_loss"], epoch)

    return patch, metrics["det_loss"], metrics["nps_loss"]
