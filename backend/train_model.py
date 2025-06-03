import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image",
                        message="Palette images with Transparency expressed in bytes should be converted to RGBA images")


# Configurations - Parameters
DATA_PATH = 'data/'
MODEL_PATH = 'sneaker_classifier.pth'
NUM_CLASSES = 50
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001


# Setting up data loading and transformations
def data_setup(data_path, batch_size):
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
        train_dataset = torch.utils.data.Subset(
            train_dataset_full, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)

        # Data Loaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
            'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        }
        dataset_sizes = {'train': len(
            train_dataset), 'test': len(test_dataset)}
        print("Dataset loaded and split successfully.")
        print(f"Training set size: {dataset_sizes['train']}")
        print(f"Test set size: {dataset_sizes['test']}\n")

        return dataloaders

    except FileNotFoundError:
        print(
            f"Error: Dataset not found in {DATA_PATH}. Please check the path.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        exit(1)


# ResNet18 Model Setup
def model_setup(NUM_CLASSES, LEARNING_RATE):
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

    # Optimizer and Loss Function set up
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    return model, criterion, optimizer, device


# Training function
def train_model(model, criterion, optimizer, dataloaders, device, num_epochs, model_path):
    print("Starting training...")
    best_weights = model.state_dict()
    best_acc = 0.0

    # Epoch loop
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1}/{num_epochs}')

        # Set train/test mode
        for mode in ['train', 'test']:
            if mode == 'train':
                model.train()
            else:
                model.eval()

            # Iteration
            running_loss = 0.0
            running_corrects = 0
            samples = 0
            epoch_size = len(dataloaders[mode])

            for i, (features, labels) in enumerate(dataloaders[mode]):
                features = features.to(device)
                labels = labels.to(device)

                # Gradient reset
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode == 'train'):
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Compute gradients and update weights
                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                # Update stats
                running_loss += loss.item() * features.size(0)
                running_corrects += torch.sum(preds == labels.data)
                samples += features.size(0)

                # Progress update
                progress_percentage = (i + 1) / epoch_size * 100
                print(
                    f'\r{mode.capitalize():<6} Batch {i+1}/{epoch_size} ({progress_percentage:.2f}%) completed.', end='')

            print()

            # Handle zero division
            if samples > 0:
                epoch_loss = running_loss / samples
                epoch_acc = running_corrects.double() / samples
                print(
                    f'{mode.capitalize()} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
            else:
                print(f'{mode.capitalize()} No samples processed in epoch')
                epoch_loss = float('nan')
                epoch_acc = float('nan')

            # Copy best model
            if mode == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = model.state_dict().copy()
                print(f'New best test accuracy: {best_acc:.4f}')

    # Save best model & weights
    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), model_path)
    print(f'Best model saved to {model_path}')
    return model


# Main
if __name__ == "__main__":
    # Data setup
    dataloaders = data_setup(DATA_PATH, BATCH_SIZE)
    model, criterion, optimizer, device = model_setup(
        NUM_CLASSES, LEARNING_RATE)
    if 'dataloaders' in locals() and 'model' in locals() and 'optimizer' in locals() and 'criterion' in locals():
        print("Setup complete. Starting model training:\n")
        train_model(model, criterion, optimizer, dataloaders,
                    device, NUM_EPOCHS, MODEL_PATH)
    else:
        print("Error occurred during setup")
        exit(1)
