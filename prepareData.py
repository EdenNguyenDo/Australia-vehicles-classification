

# !pip install timm flash_attn einops;
# !pip install -q roboflow git+https://github.com/roboflow/supervision.git

"""
This file create 2 classes with methods to prepare the image and annotation dataset for training and validating.
"""


import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
from PIL import Image



class JSONLDataset:
    """
    A dataset class for loading images and annotations from a JSONL file.

    Attributes:
        jsonl_file_path (str): Path to the JSONL file containing annotations.
        image_directory_path (str): Path to the directory containing images.
        entries (List[Dict[str, Any]]): Parsed list of annotation entries.
    """

    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        """
        Initialize the dataset with a JSONL file and image directory.

        Args:
            jsonl_file_path (str): Path to the JSONL annotation file.
            image_directory_path (str): Path to the directory containing images.
        """
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()

    def _load_entries(self) -> List[Dict[str, Any]]:
        """
        Parse the JSONL file to load annotation entries.

        Returns:
            List[Dict[str, Any]]: A list of parsed entries from the JSONL file.
        """
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries

    def __len__(self) -> int:
        """Return the number of entries in the dataset."""

        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Retrieve an image and its associated annotation.

        Args:
            idx (int): Index of the desired entry.

        Returns:
            Tuple[Image.Image, Dict[str, Any]]: The image and its annotation data.

        Raises:
            IndexError: If the index is out of range.
            FileNotFoundError: If the specified image file is not found.
        """
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")

        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")


class DetectionDataset(Dataset):
    """
    A PyTorch Dataset wrapper for handling detection datasets.

    Attributes:
        dataset (JSONLDataset): The underlying dataset object.
    """

    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        """
        Initialize the detection dataset using a JSONL file and image directory.

        Args:
            jsonl_file_path (str): Path to the JSONL annotation file.
            image_directory_path (str): Path to the directory containing images.
        """
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieve a dataset entry consisting of prefix, suffix, and image.

        Args:
            idx (int): Index of the desired entry.

        Returns:
            Tuple[str, str, Image.Image]: Prefix, suffix, and the associated image.
        """
        image, data = self.dataset[idx]
        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image



