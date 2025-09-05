import h5py
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

KERNEL_SIZE = 20

root = "/Users/propoop/Downloads/dataset"
# dataset_path = "/content/drive/MyDrive/data/dataset"
big_annotation = root + "/annotations2.json"

img_paths = []

with open(big_annotation, "r") as f:
    d = json.load(f)

images = [int(d['_via_img_metadata'][s]["filename"].split(".")[0]) for s in d['_via_img_metadata']]

for i, img in enumerate(d['_via_img_metadata']):
    data = d['_via_img_metadata'][img]
    print(f"{i} / {len(images)}")
    img_path = f"{root}/{images[i]}.png"
    ground_truth = [(region['shape_attributes']['cx'], region['shape_attributes']['cy']) for region in data['regions']]
    if i < 3:
        print(ground_truth)
    img = plt.imread(img_path)

    k = np.zeros((img.shape[0], img.shape[1]))
    for j in range(0,len(ground_truth)):
        x = int(ground_truth[j][1])
        y = int(ground_truth[j][0])
        if x < img.shape[0] and y < img.shape[1]:
            k[x, y]=1
    k = gaussian_filter(k, KERNEL_SIZE)
    new_path = f"h5_source_data/{images[i]}.h5"
    with h5py.File(new_path, 'w') as hf:
        hf['density'] = k


# 1. Generate heatmaps for everything
# 2. Select random patches from the images
# 3. Transform the corresponding heatmap patch - rotations and stretching.
# 4. Pair the image path with the heatmap patch.