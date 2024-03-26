import torch
from yolov5.patch_generation.loss import ComputeLoss
from yolov5.patch_generation.utils import get_scaled_anchors
import numpy as np
import yaml


def main():
    N = 10
    hyp = "../yolov5/models/yolov5s.yaml"
    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    print(f"hyp.keys()=={hyp.keys()}")
    print(f"hyp['anchors']=={hyp['anchors']}")
    anchors, na, nl = get_scaled_anchors(hyp["anchors"], hyp["stride"])
    hyp["na"] = na
    hyp["anchors"] = anchors
    hyp["nl"] = nl
    print(f"anchors=={anchors}")
    loss = ComputeLoss(hyp)

    return True


if __name__ == "__main__":
    main()
