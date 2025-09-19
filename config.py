import torch
from torchvision import transforms
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent
# print(f"Project root: {PROJECT_ROOT}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# device = "cpu"

# Use relative paths from project root
train_labels_csv_path = PROJECT_ROOT / "mnist_images" / "train_labels.csv"
test_labels_csv_path = PROJECT_ROOT / "mnist_images" / "test_labels.csv"
train_img_dir = PROJECT_ROOT / "mnist_images"
test_img_dir = PROJECT_ROOT / "mnist_images"
input_size = (1, 28, 28)

model_path = PROJECT_ROOT / "models" / "mnist_fully_cnn.pth"
logs_dir = PROJECT_ROOT / "logs"

# train_transforms = transforms.Compose([
#     # transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
#     transforms.Resize((28, 28)),
#     transforms.RandomRotation((-15., 15.), fill=0),
#     # transforms.CenterCrop(18),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
#     ])

train_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-10., 10.), fill=0),     # Reduce from 15° to 12°
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03),                     # Very subtle translation ±3%
            # scale=(0.95, 1.05),                         # Very subtle scaling ±5%
            fill=0
        )
    ], p=0.2),                                          # Apply only 30% of time
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.13252,), (0.31048,))
    ])

scheduler_type = ''

lr_finder_kwargs = {
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': 500,
            'step_mode': 'exp'
        }

onecycle_kwargs = {
            'lr_strategy': 'manual',  # 'conservative', 'manual'
            'pct_start': 0.1,
            'anneal_strategy': 'cos',
            'div_factor': 10.0,
            'final_div_factor': 100.0
        }