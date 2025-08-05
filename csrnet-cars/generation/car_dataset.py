import os
import json
from PIL import Image

import torch
from torchvision.transforms import v2, functional as F

from torch.utils.data import Dataset

import torch
from torchvision.transforms import functional as F
from PIL import Image

from torchvision.transforms import functional as F
from PIL import Image


class CarDataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, transform=None):
        self.root_dir = root_dir
        self.paths = []
        self.targets = []
        self.transform = None

        with open(os.path.join(annotation_file), "r") as f:
            data = json.load(f)["data"]
            for image in data:
                self.paths.append(os.path.join(root_dir, str(image["id"]) + ".png"))
                self.targets.append(image["target"])

        self.n_samples = len(self.paths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # assert index < len(self)
        path = self.paths[index]
        img = Image.open(path).convert("RGB")
        keypoints = torch.tensor([[x, y] for (x, y) in self.targets[index]], dtype=torch.float)

        if self.transform:
            img = self.transform(img)

        return img, keypoints


dataset_path = "car_data"
annotation_file = "new_annotations.json"


dataset = CarDataset(dataset_path, annotation_file)

