import torch
from yolov5.patch_generation.loss import extract_patch_grads
from yolov5.patch_generation.utils import apply_patch, init_patch


def main():
    patch = init_patch()
    image = torch.randn(3, 512, 512)
    patched_image, position = apply_patch(image, patch)
    patched_image.requires_grad_(True)
    layer = torch.nn.Conv2d(3, 3, (1, 1))
    out = layer(patched_image).sum()
    out.backward()
    print(f"patched_image.grad_fn=={patched_image.grad_fn}")
    print(f"patched_image.grad[0, position[0], position[1]]=={patched_image.grad[0, position[0], position[1]]}")
    patch_grads = extract_patch_grads(patched_image, position, patch.shape[1:])
    print(f"patch_grads.shape=={patch_grads.shape}, patch.shape=={patch.shape}, "
          f"patch_grads[0, 0, 0]=={patch_grads[0, 0, 0]}")
    assert patch_grads[0, 0, 0] == patched_image.grad[0, position[0], position[1]]
    return True


if __name__ == "__main__":
    main()
