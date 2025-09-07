import argparse
from config import device, train_labels_csv_path, test_labels_csv_path, train_img_dir, test_img_dir, train_transforms, test_transforms
from dataset import CustomMnistDataset
from torch.utils.data import DataLoader
from model import MnistFullyCNN
import torch
from torch import nn
from train import trainer
from utils import show_model_summary
from config import input_size

def main(epochs:int, lr:float, batch_size:int):
    train_dataset = CustomMnistDataset(train_labels_csv_path, train_img_dir, train_transforms)
    test_dataset = CustomMnistDataset(test_labels_csv_path, test_img_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MnistFullyCNN()
    model = model.to(device)
    show_model_summary(model, input_size)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    trainer(epochs, train_loader, test_loader, model, loss_fn, optim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and deploy MNIST CNN")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.epochs, args.lr, args.batch_size)
