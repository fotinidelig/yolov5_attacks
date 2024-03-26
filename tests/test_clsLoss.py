import torch
from yolov5.patch_generation.loss import ClassificationLoss
import numpy as np


def main():
    N = 10
    preds = torch.randn(N, 200, 85)
    gt_n_boxes = np.random.randint(1, 10, N)
    gtruths_clsid = [torch.randint(80, (gt_n_boxes[i], 1)) for i in range(N)]
    gtruths_bboxes = [torch.randn(gt_n_boxes[i], 4) for i in range(N)]
    gtruths = [torch.cat([gtruths_bboxes[i], gtruths_clsid[i]], dim=1) for i in range(10)]
    print(f"preds.shape={preds.shape}")
    print(f"[list(g.shape) for g in gtruths]={[list(g.shape) for g in gtruths]}")
    loss = ClassificationLoss(n_classes=80, iou_threshold=0)(preds, gtruths)

    print(f"loss={loss}")
    assert loss
    return True


if __name__ == "__main__":
    main()
