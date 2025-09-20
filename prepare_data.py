# prepare_data.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --- 1. Define Image Transformations ---

# Define the image size and the mean/std for normalization (standard for ImageNet models)
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Transformations for the training dataset with data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# Transformations for the validation and test datasets (no augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# --- 2. Load Datasets using ImageFolder ---

# Define the path to your main dataset directory
dataset_path = 'dataset/'

# Create the PyTorch Dataset objects
train_dataset = torchvision.datasets.ImageFolder(
    root=dataset_path + 'train',
    transform=train_transforms
)

validation_dataset = torchvision.datasets.ImageFolder(
    root=dataset_path + 'validation',
    transform=val_test_transforms
)

test_dataset = torchvision.datasets.ImageFolder(
    root=dataset_path + 'test',
    transform=val_test_transforms
)

# --- 3. Create DataLoaders ---

# Define the batch size for loading data
BATCH_SIZE = 32

# Create the DataLoader objects to feed data to the model in batches
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- 4. Verification Block ---
# This code only runs when you execute `python prepare_data.py` directly.
# It's used to test that the data is loading correctly.
if __name__ == '__main__':
    print("--- Data Loader Verification ---")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(validation_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # Check a sample batch from the train_loader
    images, labels = next(iter(train_loader))
    print(f"\nSample batch shape (Images): {images.shape}")
    print(f"Sample batch shape (Labels): {labels.shape}")