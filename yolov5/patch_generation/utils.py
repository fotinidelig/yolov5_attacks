import torch
import glob
import os.path as osp
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import json
from typing import List
import argparse
from easydict import EasyDict as edict


RESOLUTION = [256, 256]
PATCH_SIZE = [10, 10]
IMG_FORMAT = {".png", ".jpg", ".jpeg"}


def generate_patch(resolution=None, p_size=None):
    if p_size is None:
        p_size = PATCH_SIZE
    if resolution is None:
        resolution = RESOLUTION
    patch = torch.ones(3, resolution[0], resolution[1])
    for i in range(p_size[0]):
        for j in range(p_size[1]):
            patch[i, j] = float(torch.rand(1, 1))

    patch.requires_grad_(True)
    return patch


def load_config_object(cfg_path: str) -> edict:
    """
    Loads a config json and returns a edict object
    """
    with open(cfg_path, 'r', encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)

    return edict(cfg_dict)


def get_scaled_anchors(anchors: List[List[int]], stride: List[int]):
    assert len(anchors) == len(stride)
    nl = len(anchors)
    anchors_per_layer = int(len(anchors[0])/2)
    na = anchors_per_layer

    hw_anchors = []
    for i in range(nl):
        dims = [dim/stride[i] for dim in anchors[i]]
        hw_tensor = torch.stack([
            torch.tensor([dims[2*k], dims[2*k+1]]) for k in range(anchors_per_layer)
            ])
        hw_anchors.append(hw_tensor)
    return hw_anchors, nl, na


# credit: https://github.com/SamSamhuns/yolov5_adversarial/blob/master/adv_patch_gen/utils/dataset.py
class YOLODataset(Dataset):
    """Create a dataset for adversarial-yolo.

    Attributes:
        image_dir: Directory containing the images of the YOLO format dataset.
        label_dir: Directory containing the labels of the YOLO format dataset.
        max_labels: max number labels to use for each image
        model_in_sz: model input image size (height, width)
        use_even_odd_images: optionally load a data subset based on the last numeric char of the img filename [all, even, odd]
        filter_class_id: np.ndarray class id(s) to get. Set None to get all classes
        min_pixel_area: min pixel area below which all boxes are filtered out. (Out of the model in size area)
        shuffle: Whether or not to shuffle the dataset.
    """

    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 max_labels: int,
                 model_in_sz: Tuple[int, int],
                 use_even_odd_images: str = "all",
                 transform: Optional[torch.nn.Module] = None,
                 filter_class_ids: Optional[np.array] = None,
                 min_pixel_area: Optional[int] = None,
                 shuffle: bool = True):
        assert use_even_odd_images in {"all", "even", "odd"}, "use_even_odd param can only be all, even or odd"
        image_paths = glob.glob(osp.join(image_dir, "*"))
        label_paths = glob.glob(osp.join(label_dir, "*"))
        image_paths = sorted(
            [p for p in image_paths if osp.splitext(p)[-1] in IMG_FORMAT])
        label_paths = sorted(
            [p for p in label_paths if osp.splitext(p)[-1] in {".txt"}])

        # if use_even_odd_images is set, use images with even/odd numbers in the last char of their filenames
        if use_even_odd_images in {"even", "odd"}:
            rem = 0 if use_even_odd_images == "even" else 1
            image_paths = [p for p in image_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
            label_paths = [p for p in label_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
        assert len(image_paths) == len(
            label_paths), "Number of images and number of labels don't match"
        # all corresponding image and labels must exist
        for img, lab in zip(image_paths, label_paths):
            if osp.basename(img).split('.')[0] != osp.basename(lab).split('.')[0]:
                raise FileNotFoundError(
                    f"Matching image {img} or label {lab} not found")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.model_in_sz = model_in_sz
        self.shuffle = shuffle
        self.max_n_labels = max_labels
        self.transform = transform
        self.filter_class_ids = np.asarray(filter_class_ids) if filter_class_ids is not None else None
        self.min_pixel_area = min_pixel_area

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx <= len(self), "Index range error"
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # check to see if label file contains any annotation data
        label = np.loadtxt(lab_path) if osp.getsize(lab_path) else np.zeros([1, 5])
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        # sort in reverse by bbox area
        label = np.asarray(sorted(label, key=lambda annot: -annot[3] * annot[4]))
        # selectively get classes if filter_class_ids is not None
        if self.filter_class_ids is not None:
            label = label[np.isin(label[:, 0], self.filter_class_ids)]
            label = label if len(label) > 0 else np.zeros([1, 5])

        label = torch.from_numpy(label).float()
        image, label = self.pad_and_scale(image, label)
        if self.transform:
            image = self.transform(image)
            if np.random.random() < 0.5:  # rand horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if label.shape:
                    label[:, 1] = 1 - label[:, 1]
        # filter boxes by bbox area pixels compared to the model in size (640x640 by default)
        if self.min_pixel_area is not None:
            label = label[(label[:, 3] * label[:, 4]) >= (
                self.min_pixel_area / (self.model_in_sz[0] * self.model_in_sz[1]))]
            label = label if len(label) > 0 else torch.zeros([1, 5])
        image = transforms.ToTensor()(image)
        # label = self.pad_label(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """
        Pad image and adjust label
            img is a PIL image
            lab is of fmt class x_center y_center width height with normalized coords
        """
        img_w, img_h = img.size
        if img_w == img_h:
            padded_img = img
        else:
            if img_w < img_h:
                padding = (img_h - img_w) / 2
                padded_img = Image.new('RGB', (img_h, img_h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * img_w + padding) / img_h
                lab[:, [3]] = (lab[:, [3]] * img_w / img_h)
            else:
                padding = (img_w - img_h) / 2
                padded_img = Image.new('RGB', (img_w, img_w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * img_h + padding) / img_w
                lab[:, [4]] = (lab[:, [4]] * img_h / img_w)
        padded_img = transforms.Resize(self.model_in_sz)(padded_img)

        return padded_img, lab

    def pad_label(self, label: torch.Tensor) -> torch.Tensor:
        """
        Pad labels with zeros if fewer labels than max_n_labels present
        """
        pad_size = self.max_n_labels - label.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(label, (0, 0, 0, pad_size), value=0)
        else:
            padded_lab = label[:self.max_n_labels]
        return padded_lab

    @staticmethod
    def expand_img_labels(img_idx: int, label: torch.Tensor):
        """
        Expand each target object in label with img_id and return list of tuples.
        Each tuple corresponds to one target object.
        """

        expanded = torch.zeros((len(label), 6))
        expanded[:, 1:] = label
        expanded[:, 0] = torch.full((len(label),), img_idx)
        return expanded


# credit: https://github.com/SamSamhuns/yolov5_adversarial/blob/master/adv_patch_gen/utils/loss.py
class NPSLoss(nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int]):
        super(NPSLoss, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(
            triplet_scores_fpath, size), requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array
        # square root of sum of squared difference
        color_dist = (adv_patch - self.printability_array + 0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1) + 0.000001
        color_dist = torch.sqrt(color_dist)
        # use the min distance
        color_dist_prod = torch.min(color_dist, 0)[0]
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod, 0)
        nps_score = torch.sum(nps_score, 0)
        return nps_score / torch.numel(adv_patch)

    @staticmethod
    def get_printability_array(triplet_scores_fpath: str, size: Tuple[int, int, int]) -> torch.Tensor:
        """
        Get printability tensor array holding the rgb triplets (range [0,1]) loaded from triplet_scores_fpath
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
        """
        ref_triplet_list = []
        # read in reference printability triplets into a list
        with open(triplet_scores_fpath, 'r', encoding="utf-8") as f:
            for line in f:
                ref_triplet_list.append(line.strip().split(","))

        p_h, p_w = size[1:]
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack([torch.full((p_h, p_w), r),
                                          torch.full((p_h, p_w), g),
                                          torch.full((p_h, p_w), b)])
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)


def get_detection_probs(output: torch.Tensor, n_classes: int = 80):
    """
    Extract the detection probabilities during inference, i.e. the combined class and objectness score.

    output must be of the shape [batch, -1, 5 + num_cls]
    Returns:
        tensor of shape [batch, -1, 4 + num_cls] with num_cls class probabilities
    """
    # get values necessary for transformation
    assert (output.size(-1) == (5 + n_classes))

    class_confs = output[:, :, 5:5 + n_classes]  # [batch, -1, n_classes]
    obj_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1]
    obj_score = obj_score.unsqueeze(2).repeat(1, 1, n_classes)  # [batch, -1] -> [batch, -1, n_classes]

    detection_scores = obj_score * class_confs
    probs = torch.nn.Softmax(dim=2)(detection_scores)
    output_with_probs = output[:, :, :-1].clone()
    output_with_probs[:, :, 4:] = probs

    return output_with_probs


