
"""
This script is for inferring using the trained fine-tuned Florence 2 large model for image classification
"""
from PIL import Image
from timm.models import load_custom_pretrained, load_pretrained
from peft import get_peft_model
from config.config import model, configLora, processor, CHECKPOINT, DEVICE
from helpers import utils
from helpers.utils import TaskType
from transformers import AutoProcessor, AutoModelForCausalLM

CHECKPOINT = "saved_model/saved_model_test/epoch_3"

path = "dataset/train/Screenshot 2024-11-15 102741.png"
image = Image.open(path)
image_rgb = Image.open(path).convert("RGB")

# Load trained model
peft_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

utils.set_model_info(peft_model,processor)

result = utils.run_example(TaskType.OBJECT_DETECTION, image_rgb)
utils.plot_bbox(result[TaskType.OBJECT_DETECTION], image)