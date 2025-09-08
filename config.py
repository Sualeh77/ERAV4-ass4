import torch
from torchvision import transforms
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent
print(f"Project root: {PROJECT_ROOT}")

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
device = "cpu"

# Use relative paths from project root
train_labels_csv_path = PROJECT_ROOT / "mnist_images" / "train_labels.csv"
test_labels_csv_path = PROJECT_ROOT / "mnist_images" / "test_labels.csv"
train_img_dir = PROJECT_ROOT / "mnist_images"
test_img_dir = PROJECT_ROOT / "mnist_images"
input_size = (1, 28, 28)

model_path = PROJECT_ROOT / "models" / "mnist_fully_cnn.pth"

train_transforms = transforms.Compose([
    # transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    # transforms.CenterCrop(18),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13252,), (0.31048,))
    ])