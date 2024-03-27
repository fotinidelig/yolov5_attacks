import torch
import numpy
import torchvision
from torchvision.ops import nms
import torchvision.ops.boxes as box_ops
from typing import List, Union, Tuple
from yolov5.utils.loss import FocalLoss, smooth_BCE
from yolov5.utils.metrics import bbox_iou

def non_maximum_suppression(preds: List[torch.Tensor], iou_threshold: float =0.4):
    """
    Arguments:
        preds: list of Tensors, shape bs x 25200 x 85
        iou_threshold: float
    Returns:
        list with only selected preds that passed the nms.
    """
    boxes = [preds[i][:, :4] for i in range(len(preds))]
    scores = [preds[i][:, 4] for i in range(len(preds))]

    # Perform NMS
    preds_nms = []
    for i in range(len(preds)):
        selected_indices = nms(boxes[i], scores[i], iou_threshold)
        box, score = boxes[i][selected_indices], scores[i][selected_indices].unsqueeze(1)
        preds_nms_i = torch.cat([box, score, preds[i, selected_indices, 5:]], dim=1)
        preds_nms.append(preds_nms_i)

    return preds_nms


def extract_patch_grads(patched_img_grads: torch.Tensor,
                        position: Union[List[int], Tuple[int]],
                        patch_size: Tuple[int]):
    """
    Extracts gradients only for patch which is located at `position` in patched_image.
    """
    x, y = position
    patch_grads = patched_img_grads[:, x:x+patch_size[0], y:y+patch_size[1]]
    return patch_grads


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, hyp, device="cuda"):
        """Initializes ComputeLoss with model.
        Note: autobalance is removed"""
        self.hyp = hyp  # hyperparameters

        # Define criteria
        BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["cls_pw"]], device=device))
        BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=hyp.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = hyp["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(hyp["nl"], [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = 0
        self.BCEcls, self.BCEobj, self.gr = BCEcls, BCEobj, 1.0
        self.na = hyp["na"]  # number of anchors
        self.nc = hyp["nc"]  # number of classes
        self.nl = hyp["nl"]  # number of layers
        self.anchors = hyp["anchors"].to(device)
        self.device = device


    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
        indices, and anchors.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


# credit: https://github.com/SamSamhuns/yolov5_adversarial/blob/master/adv_patch_gen/utils/loss.py
class NPSLoss(torch.nn.Module):
    """NMSLoss: calculates the non-printability-score loss of a patch.
    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.
    However, a summation of the differences is used instead of the total product to calc the NPSLoss
    Reference: https://users.ece.cmu.edu/~lbauer/papers/2016/ccs2016-face-recognition.pdf
        Args:
            triplet_scores_fpath: str, path to csv file with RGB triplets sep by commas in newlines
            size: Tuple[int, int], Tuple with height, width of the patch
    """

    def __init__(self, triplet_scores_fpath: str, size: Tuple[int, int], device: str="cuda"):
        super(NPSLoss, self).__init__()
        self.printability_array = torch.nn.Parameter(self.get_printability_array(
            triplet_scores_fpath, size), requires_grad=False).to(device)

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
    def get_printability_array(triplet_scores_fpath: str, size: Tuple[int, int]) -> torch.Tensor:
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

        p_h, p_w = size
        printability_array = []
        for ref_triplet in ref_triplet_list:
            r, g, b = map(float, ref_triplet)
            ref_tensor_img = torch.stack([torch.full((p_h, p_w), r),
                                          torch.full((p_h, p_w), g),
                                          torch.full((p_h, p_w), b)])
            printability_array.append(ref_tensor_img.float())
        return torch.stack(printability_array)


class ClassificationLoss(torch.nn.Module):
    def __init__(self, n_classes=80, iou_threshold=0.4):
        super(ClassificationLoss).__init__()
        self.n_classes = n_classes
        self.iou_threshold = iou_threshold

    def __call__(self, preds, gtruths):
        """
        Arguments:
            preds: list[torch.Tensor], Predicted bounding boxes and class scores. (bs x 25200 x 85)
            gtruths: list[torch.Tensor], Ground truth bounding boxes and class ids. (bs x n_boxes x class_id)

        Returns:
            torch.Tensor: Classification loss.
        """
        bs = len(preds)
        losses = []
        for i in range(bs):
            pred_boxes = preds[i][:, :4]
            pred_scores = preds[i][:, 5:]
            gt_boxes = gtruths[i][:, :4]
            gt_class_ids = gtruths[i][:, 4].long()

            iou = box_ops.box_iou(pred_boxes, gt_boxes)
            iou_max, gt_index = torch.max(iou, dim=1)
            valid_pred_indices = iou_max >= self.iou_threshold
            valid_gt_index = gt_index[valid_pred_indices]

            # Compute classification loss only for valid predictions
            valid_pred_scores = pred_scores[valid_pred_indices]
            classification_loss = torch.nn.functional.cross_entropy(
                valid_pred_scores,
                gt_class_ids[valid_gt_index],
                reduction='mean')
            losses.append(classification_loss)

        # Compute classification loss
        total_loss = torch.mean(torch.stack(losses))
        return total_loss


class RegressionLoss(torch.nn.Module):
    def __init__(self):
        super(RegressionLoss).__init__()

    def smooth_l1_loss(self, preds, gtruths, beta=1.0):
        """
        Compute Smooth L1 Loss based on predictions and ground truths xywh.

        Args:
            preds (torch.Tensor): Predicted values, shape bs x 4.
            gtruths (torch.Tensor): Target values, shape bs x 4.
            beta (float): Smoothing parameter.

        Returns:
            torch.Tensor: Smooth L1 loss.
        """
        diff = torch.abs(preds - gtruths)
        smooth_l1_loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return smooth_l1_loss

    def calculate_regression_loss(self, preds, gtruths):
        """
        Calculate the Complete IOU (Intersection over Union) loss for bounding box regression.

        Args:
            preds (torch.Tensor): Predicted bounding box coordinates (x, y, w, h).
            gtruths (torch.Tensor): Ground truth bounding box coordinates (x, y, w, h).

        Returns:
            torch.Tensor: Regression loss.
        """
        # Smooth L1 loss for bounding box coordinates
        regression_loss = self.smooth_l1_loss(preds, gtruths)

        # Calculate IoU between predicted and target boxes using torchvision.ops.boxes.box_iou
        iou = box_ops.box_iou(preds, gtruths)

        # Complete IOU loss (combination of IoU and smooth L1 loss)
        complete_iou_loss = (1 - iou) + regression_loss

        return complete_iou_loss

    def build_targets(self, anchors, targets, grid_size):
        """
        Build targets for object detection training.
        Args:
            anchors (torch.Tensor): Anchor boxes for each prediction layer.
            targets (torch.Tensor): Ground truth bounding box coordinates (x, y, w, h).
            grid_size (int): Size of the output grid (number of grid cells along one dimension).

        Returns:
            tuple: Tuple containing target class labels, target bounding box coordinates, and
            anchor indices for each prediction layer.
        """






