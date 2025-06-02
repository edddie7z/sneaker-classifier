import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import numpy as np

# Configurations - Parameters
DATA_PATH = 'data/'
MODEL_PATH = 'sneaker_classifier.pth'
NUM_CLASSES = 50
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001

# Data Augmentation/Transformation
# Training set will include data augmentation and normalization
# Test set will only include normalization
data_transforms = {
    'train': transforms.Compose([
        # Augmentation
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # Normalization values based from ImageNet dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
}

# Data checks
print(f"Looking for data in {os.path.abspath(DATA_PATH)}")
if not os.path.exists(DATA_PATH):
    print(f"Data path {DATA_PATH} does not exist.")
    exit(1)

# Load datasets
print("Loading and splitting datasets")
try:
    # Creating datasets
    train_dataset_full = datasets.ImageFolder(
        DATA_PATH, data_transforms['train'])
    test_dataset_full = datasets.ImageFolder(
        DATA_PATH, data_transforms['test'])

    # Get class names
    class_names = train_dataset_full.classes
    if len(class_names) != NUM_CLASSES:  # Error handling
        print(
            f"Warning: Expected {NUM_CLASSES} classes, found {len(class_names)} classes.")

    # Calculating split sizes
    dataset_size = len(train_dataset_full)
    test_size = int(0.2 * dataset_size)  # 20% data for testing
    train_size = dataset_size - test_size
    print(
        f"Total images: {dataset_size}, Training images: {train_size}, Validation images: {test_size}")

    # Splitting into actual train and test datasets
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

    # Data Loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
    print("Dataset loaded and split successfully.")
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Test set size: {dataset_sizes['test']}")

except FileNotFoundError:
    print(f"Error: Dataset not found in {DATA_PATH}. Please check the path.")
    exit(1)
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit(1)

# ResNet18 Model Setup
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False
# Modify final layer for our number of classes
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
print("ResNet18 model initialized and setup.")
print(f"Model's final layer: {model.fc}")
