import h5py
from pathlib import Path
import os

# Paths
png_dir = Path("data/car/car_data")
h5_dir = Path("data/car/car_heatmaps")

# 1. Collect density sums
densities = []
for h5_path in sorted(h5_dir.glob("*.h5"), key=lambda p: int(p.stem)):
    with h5py.File(h5_path, 'r') as f:
        matrix = f['density'][:]
        densities.append((int(h5_path.stem), matrix.sum()))

# 2. Identify zero-car files
zero_car = [idx for idx, dsum in densities if (0 <= dsum <= 3)]
print(zero_car)
if len(zero_car) > 250:
    # Remove enough zero-car files to keep only 250
    to_remove = zero_car[250:]
else:
    to_remove = []

# 3. Delete extra zero-car files (both .png and .h5)
for idx in to_remove:
    png_file = png_dir / f"{idx}.png"
    h5_file = h5_dir / f"{idx}.h5"
    if png_file.exists():
        png_file.unlink()
    if h5_file.exists():
        h5_file.unlink()

# 4. Renumber remaining files
# Get remaining indices in order
remaining_indices = sorted(int(p.stem) for p in h5_dir.glob("*.h5"))

# Temporary rename to avoid collisions
for new_idx, old_idx in enumerate(remaining_indices):
    tmp_png = png_dir / f"tmp_{new_idx}.png"
    tmp_h5 = h5_dir / f"tmp_{new_idx}.h5"
    (png_dir / f"{old_idx}.png").rename(tmp_png)
    (h5_dir / f"{old_idx}.h5").rename(tmp_h5)

# Final rename
for tmp_file in png_dir.glob("tmp_*.png"):
    new_idx = tmp_file.stem.split("_")[1]
    tmp_file.rename(png_dir / f"{new_idx}.png")

for tmp_file in h5_dir.glob("tmp_*.h5"):
    new_idx = tmp_file.stem.split("_")[1]
    tmp_file.rename(h5_dir / f"{new_idx}.h5")

print("Done! Files cleaned and renumbered.")
