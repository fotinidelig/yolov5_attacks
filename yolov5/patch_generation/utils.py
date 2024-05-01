import os
import torch
import glob
import yaml
import torchvision
import os.path as osp
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from typing import List
from easydict import EasyDict as edict


RESOLUTION = [256, 256]
PATCH_SIZE = [3, 10, 10]
IMG_FORMAT = {".png", ".jpg", ".jpeg"}


def load_config_object(cfg_path: str) -> edict:
    """
    Loads a config json and returns a edict object
    """
    with open(cfg_path, "r", encoding="utf-8") as json_file:
        cfg_dict = json.load(json_file)

    return edict(cfg_dict)


def get_scaled_anchors(anchors: List[List[int]], stride: List[int]):
    assert len(anchors) == len(stride)
    nl = len(anchors)
    anchors_per_layer = int(len(anchors[0]) / 2)
    na = anchors_per_layer

    hw_anchors = []
    for i in range(nl):
        dims = [dim / stride[i] for dim in anchors[i]]
        hw_tensor = torch.stack(
            [
                torch.tensor([dims[2 * k], dims[2 * k + 1]])
                for k in range(anchors_per_layer)
            ]
        )
        hw_anchors.append(hw_tensor)
    return torch.stack(hw_anchors), nl, na


def load_hyp_to_dict(hypfile: str):
    with open(hypfile, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    anchors, nl, na = get_scaled_anchors(hyp["anchors"], hyp["stride"])
    hyp.update({"na": na, "anchors": anchors, "nl": nl})
    return hyp


def check_or_create_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
    return


class PatchTransform(torch.nn.Module):
    """Impelments batch_size transformations over an original patch.
    Available transformations:
        change of contrast
        change of brightness
    """

    def __init__(self, patch: torch.Tensor, device: str) -> None:
        super().__init__()
        self.patch = patch.to(device)
        self.patch_size = patch.shape
        self.device = device

    def forward(self, batch_size):
        def uniform_unsqueeze_expand(min_, max_):
            t = torch.FloatTensor(batch_size).uniform_(min_, max_).to(self.device)
            t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            t = t.expand(
                batch_size, self.patch_size[0], self.patch_size[1], self.patch_size[2]
            )
            return t

        brightness = uniform_unsqueeze_expand(-0.1, 0.1)
        contrast = uniform_unsqueeze_expand(0.8, 1.2)

        transformed_patches = self.patch.repeat(batch_size, 1, 1, 1)
        transformed_patches = transformed_patches * contrast + brightness
        transformed_patches = torch.clamp(transformed_patches, 0.000001, 0.99999)

        return transformed_patches


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
        shuffle: Whether to shuffle the dataset.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        max_labels: int,
        patch_is_applied: bool,
        model_in_sz: Tuple[int, int],
        use_even_odd_images: str = "all",
        transform: Optional[torch.nn.Module] = None,
        filter_class_ids: Optional[np.array] = None,
        min_pixel_area: Optional[int] = None,
        shuffle: bool = True,
    ):
        assert use_even_odd_images in {
            "all",
            "even",
            "odd",
        }, "use_even_odd param can only be all, even or odd"
        image_paths = glob.glob(osp.join(image_dir, "*"))
        label_paths = glob.glob(osp.join(label_dir, "*"))
        image_paths = sorted(
            [p for p in image_paths if osp.splitext(p)[-1] in IMG_FORMAT]
        )
        label_paths = sorted(
            [p for p in label_paths if osp.splitext(p)[-1] in {".txt"}]
        )

        # if use_even_odd_images is set, use images with even/odd numbers in the last char of their filenames
        if use_even_odd_images in {"even", "odd"}:
            rem = 0 if use_even_odd_images == "even" else 1
            image_paths = [
                p for p in image_paths if int(osp.splitext(p)[0][-1]) % 2 == rem
            ]
            label_paths = [
                p for p in label_paths if int(osp.splitext(p)[0][-1]) % 2 == rem
            ]
        assert len(image_paths) == len(
            label_paths
        ), "Number of images and number of labels don't match"
        # all corresponding image and labels must exist
        for img, lab in zip(image_paths, label_paths):
            if osp.basename(img).split(".")[0] != osp.basename(lab).split(".")[0]:
                raise FileNotFoundError(
                    f"Matching image {img} or label {lab} not found"
                )
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.model_in_sz = model_in_sz
        self.shuffle = shuffle
        self.max_n_labels = max_labels
        self.patch_is_applied = patch_is_applied
        self.transform = transform
        self.filter_class_ids = (
            np.asarray(filter_class_ids) if filter_class_ids is not None else None
        )
        self.min_pixel_area = min_pixel_area

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx <= len(self), "Index range error"
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]
        image = Image.open(img_path).convert("RGB")
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
        if not self.patch_is_applied:  # else do padding only after applying patch
            image, label = self.pad_and_scale(image, label, self.model_in_sz)
        if self.transform:
            image = self.transform(image)
            if np.random.random() < 0.5:  # rand horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if label.shape:
                    label[:, 1] = 1 - label[:, 1]
        # filter boxes by bbox area pixels compared to the model in size (640x640 by default)
        if self.min_pixel_area is not None:
            label = label[
                (label[:, 3] * label[:, 4])
                >= (self.min_pixel_area / (self.model_in_sz[0] * self.model_in_sz[1]))
            ]
            label = label if len(label) > 0 else torch.zeros([1, 5])
        image = transforms.ToTensor()(image)
        return image, label

    # def pad_label(self, label: torch.Tensor) -> torch.Tensor:
    #     """
    #     Pad labels with zeros if fewer labels than max_n_labels present
    #     """
    #     pad_size = self.max_n_labels - label.shape[0]
    #     if pad_size > 0:
    #         padded_lab = F.pad(label, (0, 0, 0, pad_size), value=0)
    #     else:
    #         padded_lab = label[:self.max_n_labels]
    #     return padded_lab

    def collate_fn(self, batch) -> Tuple[torch.Tensor, List]:
        images, labels = [b[0] for b in batch], [b[1] for b in batch]
        if not self.patch_is_applied:
            images = torch.stack(images)
        return images, labels

    def pad_and_scale(
        self, img: PIL.Image.Image, lab: List
    ) -> Tuple[PIL.Image.Image, List]:
        """
        Pad image and adjust label
            img is a PIL image or torch.Tensor
            lab is of fmt class x_center y_center width height with normalized coords
        """
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        img_w, img_h = img.size
        if img_w == img_h:
            padded_img = img
        else:
            if img_w < img_h:
                padding = (img_h - img_w) / 2
                padded_img = Image.new("RGB", (img_h, img_h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * img_w + padding) / img_h
                lab[:, [3]] = lab[:, [3]] * img_w / img_h
            else:
                padding = (img_w - img_h) / 2
                padded_img = Image.new("RGB", (img_w, img_w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * img_h + padding) / img_w
                lab[:, [4]] = lab[:, [4]] * img_h / img_w
        padded_img = transforms.Resize(self.model_in_sz)(padded_img)

        return padded_img, lab

    def pad_and_scale_batch(self, images, labels) -> Tuple[torch.Tensor, List]:
        padd_images, padd_labels = [], []
        for img, label in zip(images, labels):
            padd_img, padd_label = self.pad_and_scale(img, label)
            padd_images.append(transforms.ToTensor()(padd_img))
            padd_labels.append(padd_label)

        return torch.stack(padd_images), padd_labels

    @staticmethod
    def expand_img_labels(img_idx: int, label: torch.Tensor):
        """
        Expand each target object in label with img_id and return list of tuples.
        Each tuple corresponds to one target object.
        Each tuple contains following: (img_idx, cls_id, x, y, w, h)
        """

        expanded = torch.zeros((len(label), len(label[0]) + 1))
        expanded[:, 1:] = label
        expanded[:, 0] = torch.full((len(label),), img_idx)
        return expanded

    @staticmethod
    def labels_to_targets(labels) -> torch.Tensor:
        """
        Creates targets according to expand_img_labels for a batch of labels
        """
        targets = list(
            map(
                lambda tup: YOLODataset.expand_img_labels(tup[0], tup[1]),
                [(i, labels[i]) for i in range(len(labels))],
            )
        )
        return torch.concat(targets)


def get_detection_probs(output: torch.Tensor, n_classes: int = 80):
    """
    Extract the detection probabilities during inference, i.e. the combined class and objectness score.

    output must be of the shape [batch, -1, 5 + num_cls]
    Returns:
        tensor of shape [batch, -1, 4 + num_cls] with num_cls class probabilities
    """
    # get values necessary for transformation
    assert output.size(-1) == (5 + n_classes)

    class_confs = output[:, :, 5 : 5 + n_classes]  # [batch, -1, n_classes]
    obj_score = output[:, :, 4]  # [batch, -1, 5 + num_cls] -> [batch, -1]
    obj_score = obj_score.unsqueeze(2).repeat(
        1, 1, n_classes
    )  # [batch, -1] -> [batch, -1, n_classes]

    detection_scores = obj_score * class_confs
    probs = torch.nn.Softmax(dim=2)(detection_scores)
    output_with_probs = output[:, :, :-1].clone()
    output_with_probs[:, :, 4:] = probs

    return output_with_probs


## Patch utils ##


def init_patch(p_size=PATCH_SIZE, p_file=None):
    if p_file:
        patch = torch.load(p_file)
        return patch
    patch = torch.rand(p_size)
    return patch


def register_numerical_hooks(optim_tensor: torch.Tensor):
    """See solution for NaN gradients:
    https://github.com/UnglvKitDe/yolov5-1/commit/ae398f7a134c7e8daed1fc623bf409c46481b635?diff=split&w=0
    """
    optim_tensor.register_hook(
        lambda grad: torch.nan_to_num(grad, nan=0.0, neginf=0.0, posinf=0.0)
    )
    return


def get_random_position(img_size=RESOLUTION, p_size=PATCH_SIZE):
    assert img_size[0] > p_size[0] and img_size[1] > p_size[1]
    h_limit = (0, img_size[0] - p_size[0])
    w_limit = (0, img_size[1] - p_size[1])
    x = np.random.randint(h_limit[0], h_limit[1], 1)[0]
    y = np.random.randint(w_limit[0], w_limit[1], 1)[0]
    return torch.tensor([x, y])


def apply_patch(image, patch, position=None):
    """
    Apply a patch to an image at a specified position.

    Args:
        image (torch.Tensor): The input image tensor.
        patch (torch.Tensor): The patch tensor to be applied.
        position (tuple): The position (x, y) where the patch should be placed.

    Returns:
        torch.Tensor: The resulting image tensor after applying the patch.
        tuple: Position within image where patch was placed.
    """
    if not position:
        for i in range(10):  # max. number of tries
            try:
                position = get_random_position(image.shape[1:], patch.shape[1:])
                assert image.shape[1] > position[0] + patch.shape[1]
                assert image.shape[2] > position[1] + patch.shape[2]
                break
            except AssertionError:
                pass
    patched_img = image.clone().detach()
    x, y = position
    patched_img[:, x : x + patch.shape[1], y : y + patch.shape[2]] = (
        patch.clone().detach()
    )

    return patched_img.float(), position


def batch_apply_patch(
    images: Union[torch.Tensor, List], patches, positions=None
) -> List:
    """
    Apply patches to images at specified (or random) positions.

    Args:
        images (torch.Tensor or List of tensors): Batch of input image tensors (shape if tensor: [batch_size, channels, height, width]).
        patches (torch.Tensor): Batch of patch tensors to be applied (shape: [batch_size, channels, patch_height, patch_width]).
        positions (list of tuples): List of positions (x, y) where each patch should be placed.
            If None, random positions will be generated for each image.

    Returns:
        List of tensors: Batch of resulting image tensors after applying the patches.
        List of tuples: Positions where each patch was placed.
    """
    patched_images = []
    new_positions = []

    # Iterate over each image-patch pair in the batch
    for i in range(len(images)):
        position_i = positions[i] if positions else None
        patched_image, position = apply_patch(images[i], patches[i], position_i)
        patched_images.append(patched_image)
        new_positions.append(position)

    return patched_images, new_positions


def apply_patch_and_pad_batch(
    dataset: YOLODataset,
    images: Union[torch.Tensor, PIL.Image.Image],
    patches: torch.Tensor,
    labels: List,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Receives batches from a YOLODataset and applies patch before padding.
    Ensures that patch is applied on the image instead of the padding.

    Args:
        dataset (YOLODataset):
        images (Union[torch.Tensor, PIL.Image.Image]):
        patches (torch.Tensor):
        labels (List):
        device (str):

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
    """
    patched_inputs, patch_positions = batch_apply_patch(images, patches)
    patched_inputs, labels = dataset.pad_and_scale_batch(patched_inputs, labels)
    patched_inputs = patched_inputs.to(device)
    targets = dataset.labels_to_targets(labels).to(device)
    return patched_inputs, targets, patch_positions


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


def save_patch(patch, file_or_path_name):
    torch.save(patch, f"{file_or_path_name}.pt")
    plt.imshow(patch.permute(1, 2, 0))
    plt.tight_layout()
    plt.axis(False)
    plt.savefig(f"{file_or_path_name}.svg")


def get_ground_truth(
    patch_size, position, img_size, n_labels=80, class_id=-1, probs=False
):
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
    center_x = (position[0] + patch_size[0] // 2) / H
    center_y = (position[1] + patch_size[1] // 2) / W
    box_h = patch_size[0] / H
    box_w = patch_size[0] / W
    if probs:
        probs = [0] * n_labels
        probs[class_id] = 1
        return [center_x, center_y, box_h, box_w] + probs
    return [class_id, center_x, center_y, box_h, box_w]
