# Load and preprocess image data 

# Loading Image Data with a custom Dataset(More Generally)
import os
import random

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Make function to find classes in target directory
def find_classes(directory):
    """
    Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}")
    # Create a dictory of index labels(computers perfer numerical rather than string label)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

# Write a custom dataset class(inherits from torch.utils.data.Dataset)
class ImageFolderCustom(Dataset):
    # Initialize with a targ_dir and transform(optional) parameter
    def __init__(self, targ_dir, transform = None):
        # Get all image paths
        self.paths = list(Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    # Make function to load images
    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)
    # override the __len__() method(optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self):
        # Return the total number of samples
        return len(self.paths)
    # override the __getitem__() method(required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index):
        # Returns one sample of data ===> features and label(X,y)
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        # Transform image if necessary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx   
          
def train_transforms():        
  train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
  ])
  return train_transforms
def test_transforms():
  test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
  ])
  return test_transforms
