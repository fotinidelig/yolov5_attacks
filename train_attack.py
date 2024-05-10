import os
import re
import torch
from pathlib import Path
import mlflow
from torch.optim import Adam, SGD
import torch.utils
import torch.utils.data
from yolov5.models.yolo import Model
from yolov5.patch_generation.utils import init_patch
from yolov5.patch_generation import CONFIG_FILE, run_experiment, info, read_config
from yolov5.patch_generation.optimization import optimization
from yolov5.patch_generation.utils import YOLODataset, load_hyp_to_dict
import configparser


FILE_PATH = Path(os.path.abspath(__file__)).parent

# Read config parameters
config = read_config(CONFIG_FILE)

EXPERIMENT_NAME = config.get("experiment", "experiment_name")
RUN_NAME = config.get("experiment", "run_name")
MODEL_IN_SZ = (
    int(config.get("model", "model_in_sz")),
    int(config.get("model", "model_in_sz")),
)

vars = {
    "epochs": int(config.get("train", "epochs")),
    "bsize": int(config.get("train", "batch_size")),
    "lr": float(config.get("train", "lr")),
    "images": config.get("data", "images"),
    "labels": config.get("data", "labels"),
    "val_images": config.get("data", "val_images"),
    "val_labels": config.get("data", "val_labels"),
    "model_cfg": config.get("model", "model_cfg"),
    "patch_size": config.getint("train", "patch_size"),
}


def main():
    print(f"current_path=={FILE_PATH}")
    mlflow.log_artifact(CONFIG_FILE)
    mlflow.log_params(vars)  # logging
    # generator = torch.Generator().manual_seed(42)

    trainset = YOLODataset(
        image_dir=vars["images"],
        label_dir=vars["labels"],
        max_labels=80,
        transform=None,
        model_in_sz=MODEL_IN_SZ,
        patch_is_applied=True,
        shuffle=True,
    )
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.random_split(trainset, [0.02, 0.98])[0],
        batch_size=vars["bsize"],
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=trainset.collate_fn,
    )
    if vars["val_images"] != "":
        validset = YOLODataset(
            image_dir=vars["val_images"],
            label_dir=vars["val_labels"],
            max_labels=80,
            transform=None,
            model_in_sz=MODEL_IN_SZ,
            patch_is_applied=True,
            shuffle=True,
        )
        valloader = torch.utils.data.DataLoader(
            torch.utils.data.random_split(validset, [0.1, 0.9])[0],
            batch_size=vars["bsize"],
            shuffle=True,
            num_workers=1,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=validset.collate_fn,
        )
    else:
        valloader = None

    # Load hyperparameters from yaml file
    hyp = vars["model_cfg"]
    hyp = load_hyp_to_dict(hyp)

    model = Model(vars["model_cfg"], ch=3, nc=80, anchors=None)
    model = model.to("cuda")

    patch_size = (3, vars['patch_size'], vars['patch_size'])
    patch = init_patch(patch_size)
    patch = patch.to("cuda").requires_grad_(True)

    optimizer = Adam([patch], lr=vars["lr"])
    # optimizer = SGD([patch], lr=vars['lr'])

    patch, *losses = optimization(
        model, hyp, trainset, trainloader, valloader, patch, optimizer, vars["epochs"]
    )
    print(f"losses=={losses}")
    return True


if __name__ == "__main__":
    experiment_id = run_experiment(
        main, experiment_name=EXPERIMENT_NAME, experiment_kwargs=None, run_name=RUN_NAME
    )
    info(f"Experiment with ID={experiment_id} finished successfully.")
