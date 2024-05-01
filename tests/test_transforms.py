import torch
from yolov5.patch_generation.utils import PatchTransform

batch_size = 2
patch_size = (3, 4, 4)  # (channels, height, width)
patch = torch.rand(patch_size)  # Random patch

# Initialize PatchTransform instance
transform = PatchTransform(patch)

# Forward pass
transformed_patches = transform(batch_size)

# Print the transformed patches
print("Transformed Patches:")
print(transformed_patches.shape, transformed_patches)