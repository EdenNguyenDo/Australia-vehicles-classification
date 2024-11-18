import os
import torch
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt



class AugmentedImageDataset(Dataset):
    def __init__(self, image_paths, transform=None, device=None, seed=None):
        """
        Args:
            image_paths (list): List of file paths to images.
            transform (callable, optional): A function/transform to apply on each image.
            device (str, optional): Device to move the image tensors to ('cuda' or 'cpu').
            seed (int, optional): Random seed for reproducibility of augmentations.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.device = device if device is not None else 'cpu'
        self.augmentations_record = []  # To store augmentations applied
        self.seed = seed if seed is not None else random.randint(0, 10000)  # Default seed if not provided

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image from file
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Set the seed for this image to ensure reproducibility
        random.seed(self.seed + idx)  # Use the index to create a unique seed for each image
        np.random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)

        # Store the augmentations applied
        applied_augmentations = []

        # Perform augmentations
        if self.transform:
            for t in self.transform:
                image, aug_info = t(image)
                if aug_info:
                    applied_augmentations.append(aug_info)

        # Convert image to tensor and move to the specified device (GPU/CPU)
        image_tensor = transforms.ToTensor()(image).to(self.device)

        # Record the augmentations applied for this image
        self.augmentations_record.append({
            'image_path': self.image_paths[idx],
            'augmentations': applied_augmentations
        })

        return image_tensor

    def save_augmentations_record(self, filename='augmentations.json'):
        """
        Save the applied augmentations history to a JSON file for future reference.
        """
        with open(filename, 'w') as f:
            json.dump(self.augmentations_record, f, indent=4)


def random_flip(image):
    """Random horizontal flip."""
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image, 'Horizontal Flip'
    return image, None


def random_rotation(image):
    """Random rotation by up to 30 degrees."""
    angle = random.randint(-30, 30)
    image = image.rotate(angle)
    return image, f'Rotation {angle} degrees'


def random_crop(image):
    """Random crop of the image."""
    width, height = image.size
    left = random.randint(0, width // 4)
    top = random.randint(0, height // 4)
    right = width - random.randint(0, width // 4)
    bottom = height - random.randint(0, height // 4)
    image = image.crop((left, top, right, bottom))
    return image, 'Random Crop'


def normalize(image):
    """Normalize image (PyTorch standard normalization)."""
    image = transforms.ToTensor()(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = normalize(image)
    return image, 'Normalization'


def get_augmentation_pipeline():
    """Define augmentation pipeline."""
    return [random_flip, random_rotation, random_crop, normalize]


# Usage Example:

# # 1. Image paths for the dataset (replace with actual paths)
# image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Example paths
#
# # 2. Define augmentation pipeline
# augmentations = get_augmentation_pipeline()
#
# # 3. Create the dataset
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset = AugmentedImageDataset(image_paths, transform=augmentations, device=device, seed=42)
#
# # 4. Create DataLoader for batching
# data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# # 5. Iterate over the DataLoader
# for batch in data_loader:
#     print(batch.shape)  # Batch of augmented images (on GPU if available)
#
#     # Print augmentations applied for the first image in the batch
#     idx = 0  # First image in the batch
#     print(f"Augmentations applied to {dataset.augmentations_record[idx]['image_path']}:")
#     print(dataset.augmentations_record[idx]['augmentations'])
#
# # 6. Save augmentation history to a JSON file for reproducibility
# dataset.save_augmentations_record('augmentations_history.json')

# Function to plot original and augmented images
def plot_images(original_image, augmented_image, augmentations):
    """
    Plots the original image and the augmented image side by side.

    Args:
        original_image: The original PIL image.
        augmented_image: The augmented PIL image.
        augmentations: A list of strings describing the augmentations applied.
    """
    # Convert augmented image to numpy array for plotting if it's a tensor
    if isinstance(augmented_image, torch.Tensor):
        augmented_image = augmented_image.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) to (H, W, C)
        augmented_image = np.clip(augmented_image, 0, 1)  # Normalize the values between 0 and 1 for display

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Plot the augmented image
    axes[1].imshow(augmented_image)
    axes[1].set_title(f"Augmented Image\n{', '.join(augmentations)}")
    axes[1].axis('off')

    plt.show()


# Load images from a folder
folder_path = 'dataset/train'  # Replace with your folder path
image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path) if
               img_name.endswith(('.png'))]

# Define augmentation pipeline
augmentations = get_augmentation_pipeline()

# Set a random seed for reproducibility
random.seed(42)

# Choose 10 images to visualize
for i, image_path in enumerate(image_paths[:10]):  # Limiting to first 10 images
    original_image = Image.open(image_path).convert('RGB')

    # Apply augmentations
    augmented_image = original_image.copy()
    applied_augmentations = []
    for augment in augmentations:
        augmented_image, aug_name = augment(augmented_image)
        if aug_name:
            applied_augmentations.append(aug_name)

    # Plot original and augmented images
    plot_images(original_image, augmented_image, applied_augmentations)

