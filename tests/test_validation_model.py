import torch
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import get_scaled_anchors
from yolov5.patch_generation.utils import YOLODataset
from yolov5.models.yolo import Model
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.build import InfiniteDataLoader
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm

SAVE_DIR = Path("/home/deli_fo/projects/object_detection/yolov5/runs/validate/")

def main():
    hyp = "../yolov5/models/yolov5s.yaml"
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
        print()
    print(f"hyp.keys()=={hyp.keys()}")
    print(f"hyp['anchors']=={hyp['anchors']}")
    anchors, na, nl = get_scaled_anchors(hyp["anchors"], hyp["stride"])
    hyp["na"] = na
    hyp["anchors"] = anchors
    hyp["nl"] = nl
    print(f"anchors=={anchors}")
    print()
    _ = ComputeLoss(hyp)

    dataset = YOLODataset(
        image_dir="/home/deli_fo/projects/object_detection/yolov5/datasets/coco128/images/train2017",
        label_dir="/home/deli_fo/projects/object_detection/yolov5/datasets/coco128/labels/train2017",
        max_labels=80,
        model_in_sz=(640, 640),
        use_even_odd_images="all",
        transform=None,
        filter_class_ids=None,
        min_pixel_area=None)
    dataloader = InfiniteDataLoader(
        dataset,
        batch_size=10,
        collate_fn=YOLODataset.collate_fn)

    model = Model("../yolov5/models/yolov5s.yaml", ch=3, nc=80, anchors=None)
    model = model.to("cuda")
    pbar = tqdm(dataloader, desc="validation batches")
    args = dict(model='/home/deli_fo/projects/object_detection/yolov5/yolov5s.pt', 
                data="/home/deli_fo/projects/object_detection/yolov5/data/coco.yaml",
                batch=25)
    validator = DetectionValidator(args=args,
                                   save_dir=SAVE_DIR,
                                   dataloader=dataloader, pbar=pbar, )
    validator()

    return True


if __name__ == "__main__":
    main()
