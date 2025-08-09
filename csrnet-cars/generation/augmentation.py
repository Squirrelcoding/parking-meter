import random

import torchvision.datasets as dset
from torchvision.transforms import v2
from torchvision.transforms import PILToTensor, ToPILImage
import torch
import pathlib

# H, W = 416, 416


dataset_path = pathlib.Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/dataset"
annotation_file = dataset_path / "annotations2.json"


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 10.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

transforms = v2.Compose([
    PILToTensor(),
    gauss_noise_tensor,
    ToPILImage(),
    v2.RandomResize(125, 175),
    v2.ColorJitter(brightness=0.3),
])

dataset = dset.CocoDetection(dataset_path, annFile=annotation_file, transform=transforms)
print(len(dataset))

img, target = random.choice(dataset)