"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
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
from torch.utils.tensorboard import SummaryWriter

def main():
    # Set device to GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_WORKERS = 0
    BATCH_SIZE = 32
    epochs = 100
    class_names = 3
    # Create a writer with all default settings
    writer = SummaryWriter()
    # Set random seed
    random.seed(42)
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Register: Let's turn our training images(contained in train_dir) and testing images(contained in test_dir)into Dataset's using our own ImageFolderCustom class.
    train_data = ImageFolderCustom(targ_dir = train_dir, transform = Food_datasets.train_transforms())
    test_data = ImageFolderCustom(targ_dir = test_dir, transform = Food_datasets.test_transforms())


    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False, # don't need to shuffle test data
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    model = tinyVGG.TinyVGG(input_shape=3,
                                hidden_units=10, 
                                output_shape=class_names).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()


    # Put model in train mode
    model.train()

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in enumerate(train_dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        # Create empty results dictionary
        results = {"train_loss": [],
                    "train_acc": [],
                }
        # Print out what's happening
        print(
            
            f"\nEpoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
        )
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
                ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss},
                           global_step=epoch)

        # Add accuracy results to SummaryWriter
        writer.add_scalars(main_tag="Accuracy", 
                           tag_scalar_dict={"train_acc": train_acc}, 
                           global_step=epoch)
        
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))
    
    # Close the writer
    writer.close()
    
    ### End new ###
    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="checkpoints",
                    model_name="best_model.pth")

if __name__ == "__main__":
    main()