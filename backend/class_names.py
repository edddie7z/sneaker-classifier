import os
import json
from torchvision import datasets, transforms

DATA_PATH = 'data/'
CLASS_NAMES_PATH = 'class_names.json'
NUM_CLASSES = 50


# Save class names into json file
def save_class_names():
    minimal_transform = transforms.Compose([transforms.ToTensor()])

    print(f"Looking for data in {os.path.abspath(DATA_PATH)}")
    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} does not exist.")
        exit(1)

    try:
        temp_dataset = datasets.ImageFolder(
            DATA_PATH, transform=minimal_transform)
        class_names = temp_dataset.classes

        if len(class_names) != NUM_CLASSES:
            print(
                f"Warning: Expected {NUM_CLASSES} classes, found {len(class_names)} classes in {DATA_PATH}.")

        if not class_names:
            print("Error: No classes found. Check your DATA_PATH and dataset structure.")
            exit(1)

        with open(CLASS_NAMES_PATH, 'w') as f:
            json.dump(class_names, f, indent=4)
        print(f"Class names saved to {os.path.abspath(CLASS_NAMES_PATH)}")
        print(class_names)

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == '__main__':
    save_class_names()
