from PIL import Image
import numpy as np
import h5py
import cv2


def load_data(img_path, train=False):
    """
    Loads ground truth data.
    """
    img_path = img_path.replace("/home/leeyh/Downloads/Shanghai/part_B_final", "/content/ShanghaiTech/part_B")
    print(img_path)
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # Code for data augmentation went here.

    target = cv2.resize(
        target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC)*64

    return img, target
