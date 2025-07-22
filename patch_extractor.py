import torch
from torchvision import transforms
from PIL import Image

def extract_patches(img, patch_size=224, grid_size=3):
    w, h = img.size
    pw, ph = patch_size, patch_size
    step_x = (w - pw) // (grid_size - 1)
    step_y = (h - ph) // (grid_size - 1)

    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = i * step_x
            top = j * step_y
            patch = img.crop((left, top, left + pw, top + ph))
            patches.append(patch)
    return patches
