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
fine-tune-florence-2-vehicle.ipynb

This script demonstrates how to fine-tune the Florence-2 model for vehicle detection using a custom dataset.
It includes data preparation, model loading, fine-tuning with LoRA, and saving the trained model.

Modules:
    - prepareData: Provides `DetectionDataset` for handling the training and validation datasets.
    - server: Includes constants like `DEVICE` for determining the hardware to use (CPU or GPU).
    - train: Contains the `train_model` function to handle the training loop.
    - config: Includes `configLora` to configure LoRA parameters for fine-tuning.
    - utils: Provides utility functions for working with Florence-2, such as running examples and plotting results.

Original file is located at
    https://colab.research.google.com/drive/1-GAqpF8VsjM8XkVO7DBA8GLpgXGJOwxj
"""

# Set model and processor for the `utils.py` module
utils.set_model_info(model, processor)

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

BATCH_SIZE = 1
NUM_WORKERS = 0



train_dataset = DetectionDataset(
    jsonl_file_path="dataset/train/annotation.jsonl",
    image_directory_path="dataset/train/"
)
val_dataset = DetectionDataset(
    jsonl_file_path="dataset/valid/annotation.jsonl",
    image_directory_path="dataset/valid/"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

# Configure and fine-tune the LoRA model
"""
Configure the LoRA model for fine-tuning.

LoRA is a low-rank decomposition method that reduces the number of trainable parameters, optimizing memory usage and training speed.
"""
peft_model = get_peft_model(model, configLora())
peft_model.print_trainable_parameters()

# Train the model
"""
Train the Florence-2 model with LoRA on the custom dataset.

Hyperparameters:
    - EPOCHS: Number of training epochs.
    - LR: Learning rate for the optimizer.
"""
EPOCHS = 17
LR = 5e-6
train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)

# Save the trained model and processor
"""
Save the fine-tuned Florence-2 model and its processor for future inference or deployment.
"""
peft_model.save_pretrained("saved_model/ft-florence2-LORA")
processor.save_pretrained("saved_model/ft-florence2-LORA")
