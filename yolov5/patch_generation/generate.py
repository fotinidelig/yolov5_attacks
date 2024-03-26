from yolov5.patch_generation.utils import YOLODataset, NPSLoss, \
    load_config_object, get_detection_probs
from yolov5.patch_generation.loss import non_maximum_suppression
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import get_scaled_anchors
from yolov5.models.yolo import Model
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import de_parallel
from yolov5.utils.general import labels_to_class_weights
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torchvision
import torch
from PIL import Image

PATCH_SIZE = (3, 50, 50)
IMG_SIZE = (640, 640)


def init_patch(p_size=PATCH_SIZE, p_file=None):
    if p_file:
        patch = torch.load(p_file)
        return patch
    patch = torch.randn(p_size).requires_grad_(True)
    return patch


def get_random_position(img_size=IMG_SIZE, p_size=PATCH_SIZE):
    assert img_size[0] > p_size[0] and img_size[1] > p_size[1]
    h_limit = (0, img_size[0]-p_size[0])
    w_limit = (0, img_size[1] - p_size[1])
    x = np.random.randint(h_limit[0], h_limit[1], 1)[0]
    y = np.random.randint(w_limit[0], w_limit[1], 1)[0]
    return x, y


def apply_patch(image, patch, position=None):
    """
    Apply a patch to an image at a specified position.

    Args:
        image (torch.Tensor): The input image tensor.
        patch (torch.Tensor): The patch tensor to be applied.
        position (tuple): The position (x, y) where the patch should be placed.

    Returns:
        torch.Tensor: The resulting image tensor after applying the patch.
    """
    if not position:
        for i in range(10): # max. number of tries
            try:
                position = get_random_position(image.shape[1:], patch.shape[1:])
                assert image.shape[1] > position[0] + patch.shape[1]
                assert image.shape[2] > position[1] + patch.shape[2]
                break
            except AssertionError:
                pass
    patched_img = image.clone().detach()
    x, y = position
    patched_img[:, x:x + patch.shape[1], y:y + patch.shape[2]] = patch.clone().detach()
    print(f"[INFO] (apply_patch.py) Set patched_image (patch size={patch.shape[1:]}) on positions: "
          f"({x}:{x + patch.shape[1]}, {y}:{y + patch.shape[1]})\n")

    return patched_img.float(), position


def get_ground_truth(patch_size, position, img_size,
                     n_labels=80, class_id=-1, probs=False):
    """
        Construct the detection ground truth for the patch attack.
        For detection, the labels are in format (clas_id, center_x, center_y, box_h, box_w)
        and normalized coordinates.
        Check https://roboflow.com/formats/yolov5-pytorch-txt?ref=ultralytics for details.

        Args:
            patch_size (tuple or torch.Size): shape of patch
            position (tuple): Position of upper left corner of patch as applied on images
            img_size (tuple or torch.Size): shape of images
            class_id (int): The target class id, if untargeted defaults to -1
            n_labels (int): number of distinct object labels
            probs (bool): whether to return tuple in the form of class probabilities (similar to model output)

        Returns:
            list: The resulting label ground truth of the patch in format:
             (clas_id, center_x, center_y, box_h, box_w)
             or
             (center_x, center_y, box_h, box_w, 0, .., 1, ..., 0)
             with class probabilities (all 0 except for class_id).
             Coordinates _x and _y are normalized by the image's width and height.
        """
    H, W = img_size
    center_x = (position[0] + patch_size[0]//2) / H
    center_y = (position[1] + patch_size[1]//2) / W
    box_h = patch_size[0] / H
    box_w = patch_size[0] / W
    if probs:
        probs = [0]*n_labels
        probs[class_id] = 1
        return [center_x, center_y, box_h, box_w] + probs
    return [class_id, center_x, center_y, box_h, box_w]


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
        batch_size=28,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False)

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

    anchors, nl, na = get_scaled_anchors(hyp["anchors"], hyp["stride"])
    hyp.update({"na": na, "anchors": anchors, "nl": nl})
    detection_loss = ComputeLoss(hyp, device="cpu")

    # Get test image
    test_image, test_label = dataset[0]
    test_targets = YOLODataset.expand_img_labels(0, test_label)
    print(f"test_label.shape=={test_label.shape}, expanded test_targets.shape=={test_targets.shape}")
    test_image = test_image.unsqueeze(0).cpu()

    print(f"Image shape: {test_image.shape}, {test_image.dtype}")

    preds = model(test_image)  # output of shape bs x 25200 x 85
    print(f"Number of outputs: len(preds)={len(preds)}, shape of outputs: preds[0].shape={preds[0].shape}",)
    total_loss, reg_obj_cls_loss = detection_loss(preds, test_targets)

    # Create patch and apply
    patch = init_patch(PATCH_SIZE)
    patched_img, position = apply_patch(test_image[0], patch)
    ground_truth = [get_ground_truth(PATCH_SIZE, IMG_SIZE, position, n_labels=80, class_id=10, probs=True)]
    nps_loss = NPSLoss("30_rgb_triplets.csv", PATCH_SIZE)

    p_loss = nps_loss(patch)
    p_loss.backward()
    print(f"Patch grad after NPSLoss backward: {patch.grad[:5,:5,:5]}")

    plt.imshow(patched_img.permute(1,2,0))
    plt.savefig("patched_img.png")
    plt.show()


if __name__ == "__main__":
    main()
