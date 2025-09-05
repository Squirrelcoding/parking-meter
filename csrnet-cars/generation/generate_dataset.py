import os
import json
import random
import h5py
import numpy as np
from PIL import Image

from torchvision.transforms import v2, functional as F
from torchvision.transforms.v2 import ToImage, ToDtype, Resize

from torch.utils.data import Dataset

def random_crop_with_heatmap(img: Image.Image, heatmap: np.ndarray, scale_range=(0.3, 0.8)):
    """
    Randomly crops a chunk of the image and corresponding heatmap section.
    """
    w, h = img.size
    heatmap_h, heatmap_w = heatmap.shape

    scale_w = random.uniform(*scale_range)
    scale_h = random.uniform(*scale_range)
    crop_w = int(w * scale_w)
    crop_h = int(h * scale_h)

    # Ensure crop fits inside image
    if crop_w >= w or crop_h >= h:
        return img, heatmap

    # Random top-left corner
    max_x = w - crop_w
    max_y = h - crop_h
    crop_x = random.randint(0, max_x)
    crop_y = random.randint(0, max_y)

    # Crop the image
    cropped_img = F.crop(img, top=crop_y, left=crop_x, height=crop_h, width=crop_w)

    # Calculate corresponding heatmap crop coordinates
    # Assuming heatmap and image have same aspect ratio but potentially different dimensions
    heatmap_scale_x = heatmap_w / w
    heatmap_scale_y = heatmap_h / h
    
    heatmap_crop_x = int(crop_x * heatmap_scale_x)
    heatmap_crop_y = int(crop_y * heatmap_scale_y)
    heatmap_crop_w = int(crop_w * heatmap_scale_x)
    heatmap_crop_h = int(crop_h * heatmap_scale_y)
    
    # Ensure heatmap crop coordinates are within bounds
    heatmap_crop_x = max(0, min(heatmap_crop_x, heatmap_w - 1))
    heatmap_crop_y = max(0, min(heatmap_crop_y, heatmap_h - 1))
    heatmap_crop_w = min(heatmap_crop_w, heatmap_w - heatmap_crop_x)
    heatmap_crop_h = min(heatmap_crop_h, heatmap_h - heatmap_crop_y)
    
    # Crop the heatmap
    cropped_heatmap = heatmap[heatmap_crop_y:heatmap_crop_y + heatmap_crop_h,
                             heatmap_crop_x:heatmap_crop_x + heatmap_crop_w]

    return cropped_img, cropped_heatmap

class CarDatasetWithH5(Dataset):
    def __init__(self, root_dir: str, h5_source_dir: str, annotation_file: str, transform=None):
        self.root_dir = root_dir
        self.h5_source_dir = h5_source_dir
        self.paths = []
        self.h5_paths = []
        self.transform = transform

        with open(os.path.join(annotation_file), "r") as f:
            data = json.load(f)["_via_img_metadata"]
            for image in data:
                img_filename = data[image]["filename"]
                self.paths.append(os.path.join(root_dir, img_filename))
                
                # Construct corresponding h5 file path
                img_basename = os.path.splitext(img_filename)[0]
                h5_filename = f"{img_basename}.h5"
                self.h5_paths.append(os.path.join(h5_source_dir, h5_filename))

        self.n_samples = len(self.paths)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        img_path = self.paths[index]
        h5_path = self.h5_paths[index]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load heatmap from h5 file
        with h5py.File(h5_path, 'r') as f:
            # Assuming the heatmap is stored under key 'density' or 'heatmap'
            # Adjust the key name based on your h5 file structure
            heatmap = f['density'][:]  # or f['heatmap'][:] or whatever key you use

        return img, heatmap

# Configuration
dataset_path = "/Users/propoop/Downloads/dataset"
# dataset_path = "/content/drive/MyDrive/data/dataset"
h5_source_path = "data/h5_source_data"  # Update this path
annotation_file = dataset_path + "/annotations2.json"
output_dir = "car/car_data"
h5_output_dir = "car/car_heatmaps"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(h5_output_dir, exist_ok=True)

# Initialize dataset
dataset = CarDatasetWithH5(dataset_path, h5_source_path, annotation_file)

data = []
j = 0
offspring_size = 50
target_size = 128

# Create a Resize transform
resize_transform = Resize(size=(target_size, target_size), antialias=True)

for idx, (img, heatmap) in enumerate(dataset):
    for i in range(offspring_size):
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            new_img, new_heatmap = random_crop_with_heatmap(img, heatmap, scale_range=(0.1, 0.5))
            
            # Check if the cropped heatmap has any cars (non-zero values)
            if np.sum(new_heatmap) > 0:  # Only keep crops with cars
                break
            attempts += 1
        
        if attempts == max_attempts:
            # If no valid crop found, use the last attempt anyway
            print(f"Warning: No cars found in crop {i} for image {idx} after {max_attempts} attempts")
        
        # Get the original size of the cropped image
        w, h = new_img.size
        original_heatmap_h, original_heatmap_w = new_heatmap.shape

        # Resize the cropped image to the target size
        new_img = resize_transform(new_img)

        # Resize the heatmap to match the target size
        # Convert to PIL Image, resize, then back to numpy
        heatmap_pil = Image.fromarray(new_heatmap.astype(np.float32))
        heatmap_resized = heatmap_pil.resize((target_size, target_size), Image.BILINEAR)
        new_heatmap_resized = np.array(heatmap_resized)
        
        # Scale the heatmap values to maintain density conservation
        scale_factor = (original_heatmap_h * original_heatmap_w) / (target_size * target_size)
        new_heatmap_resized *= scale_factor

        # Save the cropped and resized image
        img_id = offspring_size * j + i
        new_img.save(f"{output_dir}/{img_id}.png")
        
        # Save the corresponding heatmap as h5 file
        h5_filename = f"{h5_output_dir}/{img_id}.h5"
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('density', data=new_heatmap_resized)
        
        # Calculate the total count (sum of heatmap values)
        total_count = np.sum(new_heatmap_resized)
        
        data.append({
            "id": img_id,
            "count": float(total_count),  # Total number of cars in this crop
            "original_file": os.path.basename(dataset.paths[idx]).split('.')[0],
            "heatmap_file": f"{img_id}.h5"
        })
    
    j += 1
    print(f"Processed {j}/{len(dataset)} source images")

# Split the data
train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):int(len(data) * 0.8) + int(len(data) * 0.1)]
val_data = data[int(len(data) * 0.8) + int(len(data) * 0.1):]

# Save metadata
with open('car_train_data.json', 'w') as f:
    json.dump({"data": train_data}, f, indent=2)

with open('car_test_data.json', 'w') as f:
    json.dump({"data": test_data}, f, indent=2)

with open('car_val_data.json', 'w') as f:
    json.dump({"data": val_data}, f, indent=2)

with open('everything.json', 'w') as f:
    json.dump({"data": data}, f, indent=2)

print(f"Generated {len(data)} total samples:")
print(f"Train: {len(train_data)}, Test: {len(test_data)}, Val: {len(val_data)}")