from PIL import Image
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt


def load_data(img_path):
    """
    Loads image and ground truth density map.
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path, 'r')
    target = np.asarray(gt_file['density'])

    # Resize density map
    target = cv2.resize(
        target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return img, target


def save_image_and_density(img_path, output_path='output.jpg'):
    img, density = load_data(img_path)
    img_np = np.array(img)

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the image
    axes[0].imshow(img_np)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Plot the density map
    axes[1].imshow(density, cmap='jet')
    axes[1].set_title("Density Map")
    axes[1].axis('off')

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# Example usage
img_path = 'processed_IMG_105.jpg'  # Replace this
save_image_and_density(img_path, output_path='output.jpg')
