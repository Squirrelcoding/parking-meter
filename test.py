import sys
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import json

with open("everything.json", "r") as f:
    data = json.load(f)

if len(sys.argv) != 2:
    print("Usage: python main.py <number>")
    sys.exit(1)

file_number = sys.argv[1]  # This will be "X" from `python main.py X`

# Open the .h5 file
with h5py.File(f'car_heatmaps/{file_number}.h5', 'r') as f:
    matrix = f['density'][:]  # load entire dataset into memory

# print(file_number)
# print(data["data"][int(file_number)]["original_file"])

# Visualize with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(matrix, cmap=CM.jet)
plt.colorbar(label='Density')
plt.title(f'Density Heatmap ({file_number})')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
