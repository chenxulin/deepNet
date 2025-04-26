import os 
import torch
import random
import Food_datasets
import utils
from pathlib import Path
import model.TinyVGG as tinyVGG
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Food_datasets import ImageFolderCustom
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Set device to GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0
BATCH_SIZE = 16
class_names = 3

# Set random seed
random.seed(42)
# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"
# Setup train and testing paths
train_dir = image_path / "train"
test_dir = image_path / "test"

# Register: Let's turn our training images(contained in train_dir) and testing images(contained in test_dir)into Dataset's using our own ImageFolderCustom class.
test_data = ImageFolderCustom(targ_dir = test_dir, transform = Food_datasets.test_transforms())

# test_dataloader
test_dataloader = DataLoader(
      test_data,
      batch_size=BATCH_SIZE,
      shuffle=False, # don't need to shuffle test data
      num_workers=NUM_WORKERS,
      pin_memory=True,
  )
model = tinyVGG.TinyVGG(input_shape=3,
                              hidden_units=16, 
                              output_shape=class_names).to(device)
# Load the model state dict from the file
model.load_state_dict(torch.load("./checkpoints/best_model.pth"))

loss_fn = torch.nn.CrossEntropyLoss()
# Put model in eval mode
model.eval() 

# Setup test loss and test accuracy values
test_loss, test_acc = 0, 0

# Turn on inference context manager
with torch.inference_mode():
    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(test_dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        test_pred_logits = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(test_pred_logits, y)
        test_loss += loss.item()

        # Calculate and accumulate accuracy
        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

# Adjust metrics to get average loss and accuracy per batch 
test_loss = test_loss / len(test_dataloader)
test_acc = test_acc / len(test_dataloader)

print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}")