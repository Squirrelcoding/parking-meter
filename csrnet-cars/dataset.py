import os
import json
import pathlib
import random
from PIL import Image
import numpy

import torch
from torchvision.transforms import v2, functional as F
from torchvision.transforms import ToPILImage
from torchvision.transforms.v2 import ToImage, ToDtype


from torch.utils.data import Dataset

import torch
from torchvision.transforms import functional as F
from PIL import Image
import math

def rotate_image_and_keypoints(img: Image.Image, keypoints: torch.Tensor, angle: float):
    w, h = img.size
    center = torch.tensor([w / 2, h / 2])

    # Rotate image without expanding
    rotated_img = F.rotate(img, angle, expand=False)

    # If no keypoints, return early
    if keypoints.numel() == 0:
        return rotated_img, keypoints.clone()

    # Rotation matrix
    theta = math.radians(angle)
    rot_matrix = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)]
    ])

    # Rotate keypoints
    shifted = keypoints - center
    rotated = shifted @ rot_matrix.T
    rotated_keypoints = rotated + center

    # Filter out-of-bounds keypoints
    x, y = rotated_keypoints[:, 0], rotated_keypoints[:, 1]
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    filtered_keypoints = rotated_keypoints[in_bounds]

    return rotated_img, filtered_keypoints



import random
from torchvision.transforms import functional as F
from PIL import Image

def random_crop_with_keypoints(img: Image.Image, keypoints: torch.Tensor, scale_range=(0.3, 0.8)):
    """
    Randomly crops a chunk of the image and adjusts keypoints.

    Args:
        img (PIL.Image): Original image.
        keypoints (Tensor): Tensor of shape (N, 2) containing [x, y] keypoints.
        scale_range (tuple): (min_scale, max_scale) for crop size relative to original image.

    Returns:
        cropped_img (PIL.Image): Cropped image.
        cropped_keypoints (Tensor): Keypoints adjusted to the new crop. Only includes those inside the crop.
    """
    w, h = img.size

    # Choose random scale for crop
    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)
    crop_w = int(w * scale_w)
    crop_h = int(h * scale_h)

    # Ensure crop fits inside image
    if crop_w >= w or crop_h >= h:
        return img, keypoints  # no crop

    # Random top-left corner
    max_x = w - crop_w
    max_y = h - crop_h
    crop_x = random.randint(0, max_x)
    crop_y = random.randint(0, max_y)

    # Crop the image
    cropped_img = F.crop(img, top=crop_y, left=crop_x, height=crop_h, width=crop_w)

    # Filter and adjust keypoints
    kp_x = keypoints[:, 0]
    kp_y = keypoints[:, 1]

    # Keep keypoints inside crop
    inside_crop_mask = (
        (kp_x >= crop_x) & (kp_x < crop_x + crop_w) &
        (kp_y >= crop_y) & (kp_y < crop_y + crop_h)
    )

    filtered_kps = keypoints[inside_crop_mask]

    # Shift to new coordinate frame
    shifted_kps = filtered_kps - torch.tensor([crop_x, crop_y])

    return cropped_img, shifted_kps



class CarDataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.paths = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(annotation_file), "r") as f:
            data = json.load(f)["_via_img_metadata"]
            for image in data:
                self.paths.append(os.path.join(root_dir, data[image]["filename"]))
                current_regions = []
                for point in data[image]["regions"]:
                    current_regions.append((point["shape_attributes"]["cx"], point["shape_attributes"]["cy"]))
                self.targets.append(current_regions)

        self.n_samples = len(self.paths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index < len(self)
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        keypoints = torch.tensor([[x, y] for (x, y) in self.targets[index]], dtype=torch.float)

        # Rotate the image and the keypoints
        theta = random.random() * 360
        img, keypoints = rotate_image_and_keypoints(img, keypoints, theta)
        img, keypoints = random_crop_with_keypoints(img, keypoints, scale_range=(0.3, 0.5))

        return img, keypoints


transforms = v2.Compose([
    ToImage(),  # replaces PILToTensor
    v2.RandomResize(125, 175),
    v2.ColorJitter(brightness=0.3),
    ToDtype(torch.float32, scale=True),
])


target_transform = v2.Compose([
    v2.RandomResize(125, 175),  # Apply the same resizing to the target
])


dataset_path = pathlib.Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/dataset"
annotation_file = dataset_path / "annotations.json"


dataset = CarDataset(dataset_path, annotation_file, transform=transforms)


img, target = random.choice(dataset)

from PIL import ImageDraw

draw = ImageDraw.Draw(img)

for x, y in target:
    draw.ellipse((x-2, y-2, x+2, y+2), fill="red")

img.save("debug_keypoints.png")