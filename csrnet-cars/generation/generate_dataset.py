import os
import json
import pathlib
import random
from PIL import Image

import torch
from torchvision.transforms import v2, functional as F
from torchvision.transforms.v2 import ToImage, ToDtype

from torch.utils.data import Dataset

import torch
from torchvision.transforms import functional as F
from PIL import Image

import random
from torchvision.transforms import functional as F
from PIL import Image

def random_crop_with_keypoints(img: Image.Image, keypoints: torch.Tensor, scale_range=(0.3, 0.8)):
    """
    Randomly crops a chunk of the image and adjusts keypoints.
    """
    if keypoints.shape == torch.Size([0]):
        return img, keypoints

    w, h = img.size

    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)
    crop_w = int(w * scale_w)
    crop_h = int(h * scale_h)
    # print(f"CROPPING: {scale_h}, {scale_w}")

    # Ensure crop fits inside image
    if crop_w >= w or crop_h >= h:
        return img, keypoints

    # Random top-left corner
    max_x = w - crop_w
    max_y = h - crop_h
    crop_x = random.randint(0, max_x)
    crop_y = random.randint(0, max_y)

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

class RawCarDataset(Dataset):
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
        # assert index < len(self)
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        keypoints = torch.tensor([[x, y] for (x, y) in self.targets[index]], dtype=torch.float)

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

dataset = RawCarDataset(dataset_path, annotation_file, transform=transforms)

data = []

j = 0
offspring_size = 30

for img, target in dataset:
    for i in range(offspring_size):
        new_img = None
        new_target = []
        while new_target == []:
            new_img, new_target = random_crop_with_keypoints(img, target, scale_range=(0.1, 0.5))
        new_img.save(f"car_data/{offspring_size* j + i}.png")
        data.append({
            "id": offspring_size* j + i,
            "target": new_target.tolist()
        })
    j += 1
    print(f"{j}/{len(dataset)}")

train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):int(len(data) * 0.8) + int(len(data) * 0.1)]
val_data = data[int(len(data) * 0.8) + int(len(data) * 0.1):]
    
with open('car_train_data.json', 'w') as f:
    json.dump({"data": train_data}, f)

with open('car_test_data.json', 'w') as f:
    json.dump({"data": test_data}, f)

with open('car_val_data.json', 'w') as f:
    json.dump({"data": val_data}, f)

with open('everything.json', 'w') as f:
    json.dump({"data": data}, f)