import os
import yaml
import torch
import torchvision
from pathlib import Path
from torch.optim import Adam
from yolov5.models.yolo import Model
from yolov5.patch_generation.utils import init_patch
from yolov5.patch_generation.optimization import optimization
from yolov5.patch_generation.utils import YOLODataset, load_hyp_to_dict

FILE_PATH = Path(os.path.abspath(__file__)).parent

def main():
    print(f'current_path=={FILE_PATH}')
    dataset = YOLODataset(
        image_dir=FILE_PATH / ".."/ "datasets" / "coco2017" / "val2017",
        label_dir=FILE_PATH / ".."/ "datasets" / "coco2017" / "coco2017labels" / "labels" / "val2017",
        max_labels=80,
        model_in_sz=(640, 640),
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


    # Load hyperparameters from yaml file
    hyp = FILE_PATH / ".."/ "models" / "yolov5s.yaml"
    hyp = load_hyp_to_dict(hyp)

    model = Model(FILE_PATH / ".."/ "models" / "yolov5s.yaml", ch=3, nc=80, anchors=None)
    model = model.to("cuda")

    patch_size = (3, 50, 50)
    patch = init_patch(patch_size)
    patch.requires_grad_(True)
    optimizer = Adam([patch], lr=0.01)

    num_epochs = 10

    patch, det_loss, nps_loss = optimization(model, hyp, dataloader, patch, optimizer)
    print(f"det_loss=={det_loss}, nps_loss=={nps_loss}")
    return True

if __name__ == "__main__":
    main()
