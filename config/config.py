from peft import LoraConfig
from torch import device, cuda
from transformers import AutoProcessor, AutoModelForCausalLM

#todo replace below to point to the locally downloaed Florence-2-large
CHECKPOINT = "microsoft/Florence-2-large"
DEVICE = device("cuda" if cuda.is_available() else "cpu")

# Load Florence-2 model and processor
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)

def configLora():
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian"
    )
    return config


def collate_fn(batch):
    """
    Custom collate function for the DataLoader.

    Args:
        batch (list): Batch of samples containing questions, answers, and images.

    Returns:
        tuple: Processed inputs and corresponding answers.
    """
    questions, answers, images = zip(*batch)
    images = [image.convert('RGB') for image in images]
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers




