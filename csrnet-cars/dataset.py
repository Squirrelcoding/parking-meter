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
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

transforms = v2.Compose([
    PILToTensor(),
    gauss_noise_tensor,
    ToPILImage(),
    v2.RandomResize(150, 250),
    v2.RandomRotation(degrees=(0,180)),
    v2.ColorJitter(brightness=0.3),
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = dset.CocoDetection("CARPK-1/train", annFile="CARPK-1/train/coco_annotations.json", transform=transforms)

img, target = random.choice(dataset)
print(img)
img.save("output.jpg")