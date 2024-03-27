import os
import yaml
import torch
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from torch.optim import Adam
import matplotlib.pyplot as plt
from yolov5.models.yolo import Model
from yolov5.models.common import DetectMultiBackend
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import YOLODataset, \
    load_config_object, get_detection_probs, load_hyp_to_dict, \
    check_or_create_dir
from yolov5.patch_generation.optimization import optimization
from yolov5.patch_generation.loss import non_maximum_suppression, NPSLoss
from yolov5.patch_generation.utils import get_scaled_anchors, init_patch, \
    apply_patch


PATCH_SIZE = (3, 50, 50)
IMG_SIZE = (640, 640)

FILE_PATH = Path(os.path.abspath(__file__)).parent

def main(run_id: str):
    dataset = YOLODataset(
        image_dir=FILE_PATH / ".." / "datasets" / "coco2017" / "val2017",
        label_dir=FILE_PATH / ".." / "datasets" / "coco2017" / "coco2017labels" / "labels" / "val2017",
        max_labels=80,
        model_in_sz=(640, 640),
        use_even_odd_images="all",
        transform=None,
        filter_class_ids=None,
        min_pixel_area=None,
        shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=40,
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=YOLODataset.collate_fn)

    # Load hyperparameters from yaml file
    hyp = FILE_PATH / ".." / "models" / "yolov5s.yaml"
    hyp = load_hyp_to_dict(hyp)

    model = Model(FILE_PATH / ".." / "models" / "yolov5s.yaml", ch=3, nc=80, anchors=None)
    model = model.to("cuda")

    patch_size = (3, 50, 50)
    patch = init_patch(patch_size)
    patch = patch.to("cuda")
    patch.requires_grad_(True)
    optimizer = Adam([patch], lr=0.01)

    num_epochs = 10
    val_step = 2
    log_dir = FILE_PATH / ".." / "runs" / "attack" / run_id
    check_or_create_dir(log_dir)
    for ep in tqdm(range(num_epochs), desc="Epochs"):
        patch, det_loss, nps_loss = optimization(model, hyp, dataloader, patch, optimizer, device="cuda")
        if ep % val_step == 0:
            torch.save(patch, log_dir / "patch.pt")
            plt.imshow(patch.cpu().clone().detach().permute(1,2,0))
            plt.savefig(log_dir / f"patch_{ep}.svg", bbox_inches='tight')
            # evaluate
            # ...
    return True


if __name__ == "__main__":
    main(run_id="test")
