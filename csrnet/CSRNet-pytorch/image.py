import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
import cv2


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # Code for data augmentation went here.

    target = cv2.resize(
        target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC)*64

    return img, target
