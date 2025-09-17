import argparse
from config import device, train_labels_csv_path, test_labels_csv_path, train_img_dir, test_img_dir, train_transforms, test_transforms
from dataset import CustomMnistDataset
from torch.utils.data import DataLoader
from model import MnistFullyCNN
import torch
from torch import nn
from train import trainer, reset_training_history
from config import input_size
from utils import get_relative_path

def main(epochs:int, lr:float, batch_size:int, experiment_name:str):
    train_dataset = CustomMnistDataset(train_labels_csv_path, train_img_dir, train_transforms)
    test_dataset = CustomMnistDataset(test_labels_csv_path, test_img_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MnistFullyCNN()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None

    # Reset history for new training
    reset_training_history()

    # Train with evaluation every 5 epochs
    metrics, exp_dir = trainer(
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,  # Optional
        evaluate_every=10,      # Show predictions every 5 epochs
        experiment_name=experiment_name
    )

    # Access training history
    print(f"Final test accuracy: {metrics['best_accuracy']:.2f}%, Experiment logs directory: {get_relative_path(exp_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and deploy MNIST CNN")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--experiment_name", type=str, default="mnist_fully_cnn")
    args = parser.parse_args()
    main(args.epochs, args.lr, args.batch_size, args.experiment_name)
