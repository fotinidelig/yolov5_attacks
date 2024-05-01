import torch
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import get_scaled_anchors
from yolov5.patch_generation.utils import YOLODataset, \
    load_hyp_to_dict, check_or_create_dir
from yolov5.models.yolo import Model
import numpy as np
import yaml

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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        collate_fn=YOLODataset.collate_fn)

    model = Model("../yolov5/models/yolov5s.yaml", ch=3, nc=80, anchors=None)
    model = model.to("cuda")

    images, labels = next(iter(dataloader))
    targets = YOLODataset.labels_to_targets(labels).to('cuda')
    print(f"Shape targets[0]: {targets[0].shape} (im_id,cls_id,x,y,w,h), targets[1]=={targets[1]}")

    preds = model(images.to('cuda'))
    print(f"Shape len(pred): {len(preds)} (for 3 anchor resolutions), pred[2]: {preds[2].shape}, (bsize,channels,.,.,(xyhw+num_cls))")

    return True


if __name__ == "__main__":
    main()
