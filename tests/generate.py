from yolov5.patch_generation.utils import YOLODataset, \
    load_config_object, get_detection_probs
from yolov5.patch_generation.loss import non_maximum_suppression, NPSLoss
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import get_scaled_anchors, init_patch, \
    apply_patch
from yolov5.models.yolo import Model
from yolov5.models.common import DetectMultiBackend
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torchvision
import torch
from PIL import Image

PATCH_SIZE = (3, 50, 50)
IMG_SIZE = (640, 640)


def main(args=None):
    cfg = load_config_object("../../base_config.json")

    # TODO: check the configuration parameters
    dataset = YOLODataset(image_dir="../../../patch_generation/data/val2017/",
                          label_dir="../../../patch_generation/data/coco2017labels/coco/labels/val2017/",
                          max_labels=80,
                          model_in_sz=IMG_SIZE,
                          use_even_odd_images="all",
                          transform=None,
                          filter_class_ids=None,
                          min_pixel_area=None,
                          shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=YOLODataset.collate_fn)

    # Load model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # model = DetectMultiBackend(weights="../../yolov5s.pt", dnn=False,
    #                            data="../data/coco.yaml", fp16=False)

    # Load hyperparameters from yaml file
    hyp = "../models/yolov5s.yaml"
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict

    model = Model("../models/yolov5s.yaml", ch=3, nc=80, anchors=None)
    model = model.to("cpu")
    # print(f"Model keys: {model.__dict__.keys()}")

    # prepare hyper parameters

    anchors, nl, na = get_scaled_anchors(hyp["anchors"], hyp["stride"])
    hyp.update({"na": na, "anchors": anchors, "nl": nl})
    detection_loss = ComputeLoss(hyp, device="cpu")

    # Get test image
    test_image, test_label = dataset[0]
    test_targets = YOLODataset.expand_img_labels(0, test_label)
    print(f"test_label.shape=={test_label.shape}, expanded test_targets.shape=={test_targets.shape}")
    test_image = test_image.unsqueeze(0).cpu()
    for b, batch in enumerate(dataloader):
        inputs, labels = batch
        targets = list(map(lambda tup: YOLODataset.expand_img_labels(tup[0], tup[1]),
                           [(i, labels[i]) for i in range(len(labels))]))
        targets = torch.concat(targets)

    print(f"Image shape: {test_image.shape}, {test_image.dtype}")

    preds = model(test_image)  # output of shape bs x 25200 x 85
    print(f"Number of outputs: len(preds)={len(preds)}, shape of outputs: preds[0].shape={preds[0].shape}",)
    # total_loss, reg_obj_cls_loss = detection_loss(preds, test_targets)

    # Create patch and apply
    patch = init_patch(PATCH_SIZE)
    patched_img, position = apply_patch(test_image[0], patch)
    # ground_truth = [get_ground_truth(PATCH_SIZE, IMG_SIZE, position, n_labels=80, class_id=10, probs=True)]
    nps_loss = NPSLoss("30_rgb_triplets.csv", PATCH_SIZE)

    p_loss = nps_loss(patch)
    p_loss.backward()
    print(f"Patch grad after NPSLoss backward: {patch.grad[:5,:5,:5]}")

    plt.imshow(patched_img.permute(1,2,0))
    plt.savefig("patched_img.png")
    plt.show()


if __name__ == "__main__":
    main()
