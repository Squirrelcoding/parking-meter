import os
import json
import pathlib
from random import choice
from PIL import Image

import torch
import torchvision.datasets as dset
from torchvision.transforms import v2
from torchvision.transforms import PILToTensor, ToPILImage


from torch.utils.data import Dataset


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
        img = Image.open(path)
        keypoints = torch.tensor([[x, y] for (x, y) in self.targets[index]], dtype=torch.float)
        print(keypoints)

        if self.transform:
            img, keypoints = self.transform(img, keypoints)

        return img, keypoints

transforms = v2.Compose([
    PILToTensor(),
    ToPILImage(),
    v2.RandomResize(125, 175),
    v2.ColorJitter(brightness=0.3),
])

target_transform = v2.Compose([
    v2.RandomResize(125, 175),  # Apply the same resizing to the target
])


dataset_path = pathlib.Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/dataset"
annotation_file = dataset_path / "annotations.json"


dataset = CarDataset(dataset_path, annotation_file, transform=transforms, target_transform=target_transform)


img, target = choice(dataset)
print(img, target)
img.save("output.png")