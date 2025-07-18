import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from image import *
from scipy.ndimage.filters import gaussian_filter

KERNEL_SIZE = 15

#set the root to the Shanghai dataset you download
root = '/home/leeyh/Downloads/Shanghai/'

#now generate the ShanghaiA's ground truth
part_A_train = os.path.join(root,'part_A_final/train_data','images')
part_A_test = os.path.join(root,'part_A_final/test_data','images')
part_B_train = os.path.join(root,'part_B_final/train_data','images')
part_B_test = os.path.join(root,'part_B_final/test_data','images')
path_sets = [part_A_train,part_A_test]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)

    # Load matlab files and images
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','ground_truth_IMG_'))
    img = plt.imread(img_path)

    # Matrix of the same dimensions as the image containing all of the ground truth coordinates
    k = np.zeros((img.shape[0],img.shape[1]))

    # Ground truth coordinates in matlab file
    ground_truth = mat["image_info"][0,0][0,0][0]

    # Set all of the coordinates
    for i in range(0,len(ground_truth)):
        x = int(ground_truth[i][1])
        y = int(ground_truth[i][0])
        if x < img.shape[0] and y < img.shape[1]:
            k[x, y]=1

    # Apply Gaussian filter and write to an h5 file
    k = gaussian_filter(k, KERNEL_SIZE)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

# Show a sample from ShanghaiB
plt.imshow(Image.open(img_paths[0]))

ground_truth_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(ground_truth_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)
