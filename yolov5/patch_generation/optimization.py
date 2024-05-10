import os
import torch
import mlflow
import torchvision
from tqdm import tqdm
from pathlib import Path
from yolov5.patch_generation import CONFIG_FILE, check_dir, read_config
from yolov5.patch_generation.loss import ComputeLoss, NPSLoss
from yolov5.patch_generation.utils import (
    YOLODataset,
    PatchTransform,
    register_numerical_hooks,
    apply_patch_to_images,
)

FILE_PATH = Path(os.path.abspath(__file__)).parent

# Read config parameters
config = read_config(CONFIG_FILE)
VAL_STEP = int(config.get("train", "val_step"))
RUN_DIR = config.get("experiment", "run_dir")
NUM_POSITIONS = config.getint("train", "num_patch_positions")

# Check if run logging dirs exist
check_dir(RUN_DIR)
check_dir(f"{RUN_DIR}/validation/")

# TODO: fine-tune the weighting of the different losses
w_box = config.getboolean("loss", "box_loss")
w_obj = config.getboolean("loss", "obj_loss")
w_cls = config.getboolean("loss", "cls_loss")
w_det = config.getboolean("loss", "det_loss")
lambda_det = config.getboolean("loss", "lambda_det")
lambda_nps = config.getboolean("loss", "lambda_nps")

if w_det != 0 and (w_cls or w_obj or w_box):
    raise RuntimeError(f"If w_det is non-zero, all other weights should be zero.")


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
    patch: torch.Tensor to optimize with requires_grad_(True)
    targets: torch.Tensor of shape (N, 6) N number of detected objects in entire batch
    model: YOLO model
    """

    model.eval()

    # Patch transformer
    patch_transform = PatchTransform(patch, device=device)
    register_numerical_hooks(patch)  # solves errors due to NaN and inf values

    # Define loss functions
    # TODO: add more losses beside detection & non-printability loss
    detection_loss_fn = ComputeLoss(hyp, device=device)

    nps_loss_fn = NPSLoss(
        FILE_PATH / "30_rgb_triplets.csv", patch.shape[1:], device=device
    )

    ebar = tqdm(range(epochs), desc="Epochs")
    for epoch in ebar:
        patch.requires_grad_(True)

        pbar = tqdm(loader, total=len(loader), desc="Input Batches")
        # metrics = {"det_loss": 0, "nps_loss": 0, "val_det_loss": 0}
        metrics = {
            "cls_loss": 0,
            "box_loss": 0,
            "obj_loss": 0,
            "det_loss": 0,
            "nps_loss": 0,
            "val_cls_loss": 0,
            "val_box_loss": 0,
            "val_obj_loss": 0,
            "val_det_loss": 0,
        }

        for b, batch in enumerate(pbar):
            pbar.set_description(f"Input Batches ep. {epoch+1}/{ebar.total}")
            inputs, labels = batch
            inputs = inputs.to(device)

            # Patch transformation
            patch_transform() # transforms patch in-place

            # Apply patch and pad images and labels
            patched_inputs, patch_positions = apply_patch_to_images(
                patch, inputs, NUM_POSITIONS
            )
            targets = dataset.labels_to_targets(labels).to(device)
            # patched_inputs.requires_grad_(True)
            _, preds = model(patched_inputs)

            # Loss calculation
            nps_loss = nps_loss_fn(patch)
            # nps_loss.backward()

            det_loss, box_obj_cls_loss = detection_loss_fn(preds, targets)
            box_loss, obj_loss, cls_loss = box_obj_cls_loss
            balanced_loss = (
                w_det * det_loss / len(inputs)
                + w_cls * cls_loss
                + w_obj * obj_loss
                + w_box * box_loss
            )
            # balanced_loss.backward()
            loss = nps_loss + balanced_loss
            loss.backward()

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

            cls_loss, obj_loss, box_loss = (
                cls_loss.detach().item(),
                obj_loss.detach().item(),
                box_loss.detach().item(),
            )
            nps_loss = nps_loss.detach().item()
            # losses to monitor
            keys = ["box_loss", "obj_loss", "cls_loss", "nps_loss"]
            values = [box_loss, obj_loss, cls_loss, nps_loss]

            pbar.set_postfix(dict(zip(keys, values)))

            for k, v in zip(keys, values):
                metrics[k] += v

            del (
                det_loss,
                cls_loss,
                obj_loss,
                box_loss,
                nps_loss,
                patched_inputs,
            )

        for k in keys:
            metrics[k] /= b
            mlflow.log_metric(k, metrics[k], epoch)

        ebar.set_postfix(
            cls_loss=metrics["cls_loss"],
            box_loss=metrics["box_loss"],
            obj_loss=metrics["obj_loss"],
            nps_loss=metrics["nps_loss"],
        )

        if valloader is None or (
            epoch % VAL_STEP != 0 and epoch - 1 != epochs
        ):  # no validation
            continue

        # Enter validation
        torch.save(patch, f"{RUN_DIR}/validation/patch_{epoch}.pt")
        torchvision.utils.save_image(patch, f"{RUN_DIR}/validation/patch_{epoch}.png")
        pbar = tqdm(valloader, total=len(valloader), desc="Validation Batches")

        for b, batch in enumerate(pbar):
            inputs, labels = batch
            inputs = inputs.to(device)

            # Patch transformation
            patch_transform() # transforms patch in-place

            with torch.no_grad():
                patched_inputs, patch_positions = apply_patch_to_images(
                    patch, inputs, NUM_POSITIONS
                )
                targets = dataset.labels_to_targets(labels).to(device)
                _, preds = model(patched_inputs)
                det_loss, box_obj_cls_loss = detection_loss_fn(preds, targets)
            box_loss, obj_loss, cls_loss = box_obj_cls_loss

            del preds, targets, box_obj_cls_loss
            # Monitoring

            if (
                epoch == 0 and b == 0
            ):  # if first epoch and first val. batch, save images as examples
                torchvision.utils.save_image(
                    [im for im in patched_inputs],
                    f"{RUN_DIR}/validation/pathed_ims.png",
                    nrow=5,
                )

            cls_loss, obj_loss, box_loss = (
                cls_loss.detach().item(),
                obj_loss.detach().item(),
                box_loss.detach().item(),
            )
            # losses to monitor
            keys = ["val_cls_loss", "val_obj_loss", "val_box_loss"]
            values = [cls_loss, obj_loss, box_loss]

            pbar.set_postfix(dict(zip(["cls_loss", "obj_loss", "box_loss"], values)))

            for k, v in zip(keys, values):
                metrics[k] += v

            del det_loss, obj_loss, cls_loss, box_loss, patched_inputs

        for k in keys:
            metrics[k] /= b
            mlflow.log_metric(k, metrics[k], epoch)

    return (
        patch,
        metrics["box_loss"],
        metrics["obj_loss"],
        metrics["cls_loss"],
        metrics["nps_loss"],
    )
