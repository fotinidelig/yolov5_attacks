import torch
from yolov5.patch_generation.loss import non_maximum_suppression


def main():
    preds = torch.randn(10, 200, 85)
    nms_preds = non_maximum_suppression(preds)
    print(f"preds.shape={preds.shape}, len(nms_preds)={len(nms_preds)}, "
          f"nms_preds[0].shape={nms_preds[0].shape}")
    assert preds.shape[0] == len(nms_preds)
    assert preds.shape[2] == nms_preds[0].shape[1]
    return True


if __name__ == "__main__":
    main()