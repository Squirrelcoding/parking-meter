import h5py
import matplotlib.pyplot as plt
from matplotlib import cm as CM

# Open the .h5 file
with h5py.File('h5_data/0.h5', 'r') as f:
    # Access the 'density' group
    
    # Assuming the dataset inside density is called 'data'
    matrix = f['density'][:]  # load entire dataset into memory

# Visualize with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(matrix, cmap=CM.jet)  # or 'plasma', 'inferno', 'magma', etc.
plt.colorbar(label='Density')
plt.title('Density Heatmap')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()
