import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import scipy
from matplotlib import cm as CM
from image import *

# Borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(ground_truth):
    print(ground_truth.shape)
    density = np.zeros(ground_truth.shape, dtype=np.float32)
    ground_truth_count = np.count_nonzero(ground_truth)
    if ground_truth_count == 0:
        return density

    # Get the points of each of the ground truth annotations
    pts = np.array(zip(np.nonzero(ground_truth)[1], np.nonzero(ground_truth)[0]))

    # Build KDTree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)

    # Query KDTree
    distances, locations = tree.query(pts, k=4)

    print('Generating density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(ground_truth.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if ground_truth_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3])*0.1
        else:
            sigma = np.average(np.array(ground_truth.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density

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

    # Load matlab files
    mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
    img = plt.imread(img_path)

    # Create an zero matrix with the same dimensions as the image
    k = np.zeros((img.shape[0],img.shape[1]))

    # Mark ground truth coordinates
    gt = mat["image_info"][0, 0][0, 0][0]
    for i in range(0, len(gt)):
        x = int(gt[i][1])
        y = int(gt[i][0])
        if x < img.shape[0] and y < img.shape[1]:
            k[x, y] = 1

    # Apply the gaussian filter
    k = gaussian_filter_density(k)
    with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
            hf['density'] = k

# {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Nov 18 20:06:05 2016', '__version__': '1.0', '__globals__': [], 'image_info': array([[array([[(array([[ 29.6225116 , 472.92022152],
#                      [ 54.35533603, 454.96602305],
#                      [ 51.79045053, 460.46220626],
#                      ...,
#                      [597.89732076, 688.27900015],
#                      [965.77518336, 638.44693908],
#                      [166.9965574 , 628.1873971 ]], shape=(1546, 2)), array([[1546]], dtype=uint16))]],
#             dtype=[('location', 'O'), ('number', 'O')])                                                ]],
#     dtype=object)}

# Show a sample from ShanghaiA
plt.imshow(Image.open(img_paths[0]))

gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground_truth'),'r')
groundtruth = np.asarray(gt_file['density'])
plt.imshow(groundtruth,cmap=CM.jet)

