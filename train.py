import os
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from torch import no_grad
from config.config import DEVICE


def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    """
    Function to train and validate a model using a provided training and validation dataloader.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader object for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader object for validation data.
        model (transformers.PreTrainedModel): Pretrained model to be fine-tuned on the dataset.
        processor (transformers.PreTrainedProcessor): Processor (tokenizer and feature extractor)
                                                     associated with the model.
        epochs (int, optional): The number of training epochs. Defaults to 10.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-6.

    Returns:
        None. The model is trained and saved after each epoch.

    Process:
        The function performs the following steps:
        1. Initializes the optimizer (AdamW) with the model's parameters.
        2. Sets up a linear learning rate scheduler that warms up for 0 steps.
        3. Iterates through the training data for a number of epochs, updating the model weights.
        4. Computes training and validation losses and prints average loss at each epoch.
        5. Saves the model and processor after each epoch to a checkpoint directory.
    """

    # Initialize the optimizer (AdamW) with the model's parameters
    optimizer = AdamW(model.parameters(), lr=lr)

    # Total number of training steps (based on the number of epochs and batches in the training loader)
    num_training_steps = epochs * len(train_loader)

    # Initialize a linear learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,  # No warmup steps
        num_training_steps=num_training_steps,  # Total number of training steps
    )

    # Loop over the specified number of epochs
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0  # Initialize variable to track training loss

        # Training loop: iterate through training data
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            # Extract input tensors (input_ids and pixel_values) from the data
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            # Process answers to match the format expected by the model (tokenized labels)
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)  # Move to the configured device (GPU/CPU)

            # Forward pass: pass the inputs through the model
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

            # Calculate loss (cross-entropy loss by default)
            loss = outputs.loss

            # Backpropagate the gradients, optimize the model, and update the learning rate
            loss.backward()
            optimizer.step()  # Update the model's weights
            lr_scheduler.step()  # Update the learning rate
            optimizer.zero_grad()  # Zero the gradients after each step

            # Accumulate the training loss
            train_loss += loss.item()

        # Calculate and print average training loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        # Validation phase: evaluate the model on the validation set
        model.eval()  # Set model to evaluation mode
        val_loss = 0  # Initialize variable to track validation loss

        with no_grad():  # Disable gradient calculation to save memory during evaluation
            for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                # Extract input tensors (input_ids and pixel_values)
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]

                # Process answers (labels) for the validation set
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(DEVICE)  # Move to the configured device

                # Forward pass: pass the inputs through the model to get the loss
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)

                # Add the loss to the validation loss accumulator
                loss = outputs.loss
                val_loss += loss.item()

            # Calculate and print average validation loss for the epoch
            avg_val_loss = val_loss / len(val_loader)
            print(f"Average Validation Loss: {avg_val_loss}")

        # Save model checkpoints after each epoch
        output_dir = f"./model_checkpoints/epoch_{epoch + 1}"
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
        model.save_pretrained(output_dir)  # Save the model's state_dict
        processor.save_pretrained(output_dir)  # Save the processor (tokenizer/feature extractor)
