"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import ImageFolderCustom

NUM_WORKERS = os.cpu_count()
BATCH_SIZE = 32

def dataloader(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int = BATCH_SIZE, 
    num_workers: int = NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
# Register: Let's turn our training images(contained in train_dir) and testing images(contained in test_dir)into Dataset's using our own ImageFolderCustom class.
  train_data = ImageFolderCustom(targ_dir = train_dir, transform = datasets.train_transforms())
  test_data = ImageFolderCustom(targ_dir = test_dir, transform = datasets.test_transforms())

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names