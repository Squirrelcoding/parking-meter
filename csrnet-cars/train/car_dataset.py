from PIL import Image
import random

from torchvision.transforms import functional as F

from torch.utils.data import Dataset

import numpy as np
import h5py
import cv2

def load_data(img_path, train=False):
    """
    Loads ground truth data.
    """
    gt_path = img_path.replace('.png', '.h5').replace('car_data', 'h5_data')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # Code for data augmentation went here.

    target = cv2.resize(
        target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC)*64

    return img, target


class CarDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root * 4
        random.shuffle(root)
        self.n_samples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = load_data(img_path, self.train)
    
        # Code manipulating the image went here

        if self.transform is not None:
            img = self.transform(img)
        return img, target
