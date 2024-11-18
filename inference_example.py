# Import required libraries and modules
# !pip install timm flash_attn einops;
# !pip install -q roboflow git+https://github.com/roboflow/supervision.git
from prepareData import DetectionDataset
from train import train_model
from config.config import configLora, collate_fn, model, processor
from torch.utils.data import DataLoader
from peft import get_peft_model
from PIL import Image
from helpers import utils

"""
Demonstrate image loading for object detection.

Image path:
    - path: Specifies the image file path for demonstration.
"""
path = "dataset/train/frame2.png"
image = Image.open(path)
image_rgb = Image.open(path).convert("RGB")

# Perform object detection
"""
Run object detection on the sample image using the Florence-2 model.
"""
tasks = [utils.TaskType.OBJECT_DETECTION]

for task in tasks:
    results = utils.run_example(task, image_rgb)
    print(task.value)
    utils.plot_bbox(results[task], image)

# Prepare datasets and data loaders
"""
Prepare training and validation datasets and their corresponding data loaders.
"""