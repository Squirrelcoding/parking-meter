import os
import glob
import json

from torch.utils.data import Dataset


class CarDataset(Dataset):
    def __init__(self, root_dir: str, annotation_file: str, transform=None):
        self.root_dir = root_dir

        with open(os.path.join(root_dir, annotation_file), "r") as f:
            data = json.dump(f)
            self.paths = [os.path.join(root_dir, image["file_name"]) for image in data]

        self.n_samples = len(self.paths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self)

        pass
