import os
import torch
from pathlib import Path
import mlflow
from torch.optim import Adam
import torch.utils
import torch.utils.data
from yolov5.models.yolo import Model
from yolov5.patch_generation.utils import init_patch
from ultralytics.data.build import InfiniteDataLoader
from yolov5.patch_generation import CONFIG_FILE, check_dir, run_experiment, info
from yolov5.patch_generation.optimization import optimization
from yolov5.patch_generation.utils import YOLODataset, load_hyp_to_dict
import configparser


FILE_PATH = Path(os.path.abspath(__file__)).parent

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

EXPERIMENT_NAME = config.get("experiment", "experiment_name")
RUN_NAME = config.get("experiment", "run_name")
vars = {
    "epochs": int(config.get("train", "epochs")),
    "bsize": int(config.get("train", "batch_size")),
    "images": config.get("data", "images"),
    "labels": config.get("data", "labels"),
    "model_cfg": config.get("model", "model_cfg")
}

def main():
    print(f'current_path=={FILE_PATH}')
    mlflow.log_params(vars) # logging
    dataset = YOLODataset(
        image_dir=vars['images'],
        label_dir=vars['labels'],
        max_labels=80,
        model_in_sz=(640, 640),
        use_even_odd_images="all",
        transform=None,
        patch_is_applied=True,
        filter_class_ids=None,
        min_pixel_area=None,
        shuffle=True)
    generator = torch.Generator().manual_seed(42)
    
    validset, trainset = torch.utils.data.random_split(dataset, [0.2, 0.8], generator=generator)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=vars['bsize'],
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=dataset.collate_fn)

    valloader = torch.utils.data.DataLoader(
        validset,
        batch_size=vars['bsize'],
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=dataset.collate_fn)

    # Load hyperparameters from yaml file
    hyp = vars['model_cfg']
    hyp = load_hyp_to_dict(hyp)

    model = Model(vars['model_cfg'], ch=3, nc=80, anchors=None)
    model = model.to("cuda")

    patch_size = (3, 50, 50)
    patch = init_patch(patch_size)
    patch = patch.to("cuda").requires_grad_(True)
    optimizer = Adam([patch], lr=0.001)
    patch, det_loss, nps_loss = optimization(model, hyp, dataset, trainloader, valloader, patch, optimizer, vars['epochs'])
    print(f"det_loss=={det_loss}, nps_loss=={nps_loss}")
    return True


if __name__ == "__main__":
    experiment_id = run_experiment(main, experiment_name=EXPERIMENT_NAME, experiment_kwargs=None, run_name=RUN_NAME)
    info(f"Experiment with ID={experiment_id} finished successfully.")
