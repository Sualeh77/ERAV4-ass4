import argparse
from config import device, train_labels_csv_path, test_labels_csv_path, train_img_dir, test_img_dir, train_transforms, test_transforms, scheduler_type, lr_finder_kwargs, onecycle_kwargs
from dataset import CustomMnistDataset
from torch.utils.data import DataLoader
from model import MnistFullyCNN, MnistFullyCNNSmall, MnistFullyCNNSmaller, MnistFullyCNNSmaller8kParams, MnistFullyCNNTiny4kParams
import torch
from torch import nn
from train import trainer, reset_training_history
from config import input_size
from utils import get_relative_path, setup_logging
from lr_finder_utils import setup_onecycle_policy

def main(epochs:int, lr:float, batch_size:int, experiment_name:str):
    # Setup logging
    experiment_dir, experiment_start_time, logger = setup_logging(experiment_name)

    train_dataset = CustomMnistDataset(train_labels_csv_path, train_img_dir, train_transforms)
    test_dataset = CustomMnistDataset(test_labels_csv_path, test_img_dir, test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = MnistFullyCNNSmaller()
    # model = MnistFullyCNNTiny4kParams()
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Use AdamW with weight decay instead of Adam
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4,      # Add weight decay for better generalization
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    if scheduler_type == 'onecycle':
        scheduler = setup_onecycle_policy(epochs,lr_finder_kwargs, onecycle_kwargs,
        train_loader, experiment_dir, model, loss_fn, optimizer, logger)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        mode='max',           # Monitor accuracy (maximize)
                        factor=0.5,           # More aggressive reduction: LR *= 0.2 (instead of 0.5)
                        patience=2,           # Reduce patience: wait only 2 epochs (instead of 3)
                        threshold=0.001,      # Only reduce if improvement < 0.1%
                        min_lr=1e-6,          # Prevent LR from going too low
                        cooldown=0            # Wait n epoch after reduction before monitoring again
                    )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=20,
        #     eta_min=1e-6
        # )

    # Reset history for new training
    reset_training_history()

    # Train with evaluation every 5 epochs
    metrics, exp_dir = trainer(
        logger=logger,
        epochs=epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,  # Optional
        evaluate_every=20,      # Show predictions every 5 epochs
        experiment_name=experiment_name,
        experiment_dir=experiment_dir,
        experiment_start_time=experiment_start_time
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
