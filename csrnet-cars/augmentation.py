import random

import torchvision.datasets as dset
from torchvision.transforms import v2
from torchvision.transforms import PILToTensor, ToPILImage
from PIL import Image
import torch

H, W = 416, 416

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

dataset = dset.CocoDetection("CARPK-1/train", annFile="CARPK-1/train/coco_annotations.json", transform=transforms)

img, target = random.choice(dataset)
print(img)
img.save("output.jpg")