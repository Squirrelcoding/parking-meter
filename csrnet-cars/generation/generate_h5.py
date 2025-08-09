import h5py
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

KERNEL_SIZE = 3

NUM_CARS = 1530

root = "/car_data/"
big_annotation = "everything.json"

img_paths = []

with open(big_annotation) as f:
    d = json.load(f)

for i in range(NUM_CARS):
    print(f"{i} / {NUM_CARS}")
    img_path = f"car_data/{i}.png"
    ground_truth = d["data"][i]["target"]
    img = plt.imread(img_path)

    k = np.zeros((img.shape[0], img.shape[1]))
    for j in range(0,len(ground_truth)):
        x = int(ground_truth[j][1])
        y = int(ground_truth[j][0])
        if x < img.shape[0] and y < img.shape[1]:
            k[x, y]=1
    k = gaussian_filter(k, KERNEL_SIZE)
    with h5py.File(img_path.replace(".png", ".h5").replace("car_data", "h5_data"), 'w') as hf:
        hf['density'] = k