import h5py
import json

with open("everything.json", "r") as f:
    data = json.load(f)

results = []  # store (id, sum) tuples

for file_number in range(0, 2501):  # IDs from 0 to 2500 inclusive
    path = f'data/car/car_heatmaps/{file_number}.h5'
    try:
        with h5py.File(path, 'r') as f:
            matrix = f['density'][:]  # load the density map
            density_sum = matrix.sum()
            results.append((file_number, density_sum))
    except FileNotFoundError:
        print(f"Warning: file {path} not found.")
    except KeyError:
        print(f"Warning: 'density' dataset missing in {path}.")

# print results sorted by ID
for file_number, total in results:
    print(f"ID {file_number:04d}: sum = {total:.2f}")

# optionally, show top outliers
print("\nTop 10 patches by density sum:")
for file_number, total in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
    print(f"ID {file_number:04d}: sum = {total:.2f}")
