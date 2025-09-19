from config import device, logs_dir, model_path
import torch
from tqdm import tqdm
from utils import GetCorrectPredCount, log_model_info, log_dataset_info, log_training_config, save_metrics_to_json, get_relative_path
from torch import nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pathlib import Path
import torch.nn.functional as F
from vizualize import plot_training_history
import time
from metrics_tracker import MetricsTracker

# Global lists to track training history
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer, epoch: int, logger, scheduler=None) -> Tuple[float, float]:
    """
    training function
    """
    model.train()

    train_loss = 0
    processed = 0

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(num_classes=10, device=device)

    # Enhanced progress bar with more info
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')

    for batch_idx, (image, label) in enumerate(pbar):
        image, label = image.to(device), label.to(device)

        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(image)
        loss = loss_fn(pred, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Step scheduler if it's OneCycleLR (step per batch)
        # Step scheduler if it's OneCycleLR (step per batch)
        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            # Log LR changes every 100 batches for debugging
            if batch_idx % 100 == 0:
                logger.info(f"   Batch {batch_idx}: LR changed from {old_lr:.6f} to {new_lr:.6f}")

         # Update metrics
        train_loss += loss.item()
        processed += len(image)

        # Update accuracy metric
        metrics_tracker.update(pred, label)
        current_metrics = metrics_tracker.compute()
        avg_loss = train_loss / (batch_idx + 1)

        # Update progress bar with current metrics
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg_Loss': f'{avg_loss:.4f}',
            'Accuracy': f'{current_metrics["accuracy"]:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # Log batch details (every 100 batches to avoid spam)
        # if batch_idx % 100 == 0 and logger:
        #     logger.debug(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}, Acc={current_accuracy:.2f}%")
        
    # Calculate final epoch metrics
    final_loss = train_loss / len(dataloader)
    final_metrics = metrics_tracker.compute()
    final_accuracy = final_metrics["accuracy"]

    # Store for plotting
    train_losses.append(final_loss)
    train_accuracies.append(final_accuracy)

    # Log epoch results
    if logger:
        logger.info(f"📈 Epoch {epoch+1} Training Results:")
        logger.info(f"   Loss: {final_loss:.4f}")
        logger.info(f"   Accuracy: {final_accuracy:.2f}% ({processed} samples)")
        logger.info(f"   Precision: {final_metrics['precision']:.2f}%")
        logger.info(f"   Recall: {final_metrics['recall']:.2f}%")
        logger.info(f"   F1-Score: {final_metrics['f1_score']:.2f}%")
        logger.info(f"   LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Reset metric for next epoch
    metrics_tracker.reset()

    return final_loss, final_accuracy

def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, epoch: int, logger) -> Tuple[float, float]:
    """
    test function
    """
    model.eval()

    test_loss = 0
    processed = 0 

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(num_classes=10, device=device)

    # Progress bar for test
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Test]')

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(pbar):
            image, label = image.to(device), label.to(device)

            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            processed += len(image)

            # Update accuracy metric
            metrics_tracker.update(pred, label)
            current_metrics = metrics_tracker.compute()
            avg_loss = test_loss / (batch_idx + 1)

            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Accuracy': f'{current_metrics["accuracy"]:.2f}%'
            })
    
    # Calculate final metrics
    final_loss = test_loss / len(dataloader)
    final_metrics = metrics_tracker.compute()
    final_accuracy = final_metrics["accuracy"]

    # Store for plotting
    test_losses.append(final_loss)
    test_accuracies.append(final_accuracy)

    # Log results
    if logger:
        logger.info(f"📊 Epoch {epoch+1} Test Results:")
        logger.info(f"   Loss: {final_loss:.4f}")
        logger.info(f"   Accuracy: {final_accuracy:.2f}% ({processed} samples)")
        logger.info(f"   Precision: {final_metrics['precision']:.2f}%")
        logger.info(f"   Recall: {final_metrics['recall']:.2f}%")
        logger.info(f"   F1-Score: {final_metrics['f1_score']:.2f}%")
    
    # Reset metric for next epoch
    metrics_tracker.reset()

    return final_loss, final_accuracy

def evaluate_model(model: nn.Module, test_loader: DataLoader, epoch: int,
                experiment_dir: Path, logger, num_samples: int = 5):
    """
    Evaluate model and show sample predictions with images
    """
    model.eval()

    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted = outputs.argmax(dim=1)

    # Create subplot
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(f'Sample Predictions - Epoch {epoch+1}', fontsize=16, fontweight='bold')
    
    correct_predictions = 0
    
    for i in range(num_samples):
        # Convert image to numpy and denormalize
        img = images[i].cpu().squeeze().numpy()
        
        # Plot image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        # Get prediction info
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item()
        
        if true_label == pred_label:
            correct_predictions += 1
        
        # Color coding: green for correct, red for incorrect
        color = 'green' if true_label == pred_label else 'red'
        
        # Set title with prediction info
        axes[i].set_title(
            f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}',
            fontsize=10,
            color=color,
            fontweight='bold'
        )
    
    plt.tight_layout()
    
    # Save to experiment directory
    pred_image_path = experiment_dir / f'predictions_epoch_{epoch+1}.png'
    plt.savefig(pred_image_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to save memory

    # Log evaluation results
    if logger:
        logger.info(f"🔍 Epoch {epoch+1} Evaluation:")
        logger.info(f"   Sample accuracy: {correct_predictions}/{num_samples} ({100*correct_predictions/num_samples:.1f}%)")
        logger.info(f"   Predictions saved to: {get_relative_path(pred_image_path)}")
    
    # Print detailed predictions
    print(f"🔍 Detailed Predictions (Epoch {epoch+1}):")
    print("-" * 60)
    for i in range(num_samples):
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item()
        status = "✅ Correct" if true_label == pred_label else "❌ Wrong"

        # Log individual predictions
        if logger:
            logger.debug(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Conf={confidence:.3f}, Status={status}")
    print("-" * 60)

def trainer(logger, epochs: int, train_loader: DataLoader, test_loader: DataLoader, 
           model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
           scheduler=None, evaluate_every: int = 20, experiment_name: str = None,
           experiment_dir=logs_dir, experiment_start_time=None):
    """
    Trainer function
    """
    # Log all configuration info
    log_model_info(logger, model)
    log_dataset_info(logger, train_loader, test_loader)
    log_training_config(logger, optimizer, loss_fn, scheduler)

    logger.info(f"🎯 Training Configuration:")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Evaluate every: {evaluate_every} epochs")
    logger.info("=" * 80)

    logger.info("🚀 Starting Training Process...")
    logger.info(f"⚙️  Device: {device}")
    logger.info("=" * 80)

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        logger.info(f"🔄 Starting Epoch {epoch+1}/{epochs}")
        logger.info("-" * 50)
        
         # Training phase
        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, epoch, logger)

        # Testing phase
        test_loss, test_acc = test(test_loader, model, loss_fn, epoch, logger)

        # Update learning rate scheduler if provided (but NOT for OneCycleLR - it's already stepped per batch)
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            old_lr = optimizer.param_groups[0]['lr']
            
            # ReduceLROnPlateau needs the metric to monitor
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_acc)  # Pass test accuracy for monitoring
            else:
                scheduler.step()  # Other schedulers don't need metrics
                
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                logger.info(f"📉 Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")

         # Track best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            # Save best model
            best_model_path = model_path.parent / f"{model_path.stem}_best{model_path.suffix}"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"🏆 New best model saved! Accuracy: {best_accuracy:.2f}%")

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time

         # Log epoch summary
        logger.info(f"📈 Epoch {epoch+1} Summary:")
        logger.info(f"   Duration: {epoch_duration:.2f}s")
        logger.info(f"   Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        logger.info(f"   Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        logger.info(f"   🏆 Best Test Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")

        # Evaluate model with sample predictions every N epochs
        if (epoch + 1) % evaluate_every == 0 or epoch == epochs - 1:
            print(f"\n🔍 Evaluation at Epoch {epoch+1}:")
            evaluate_model(model, test_loader, epoch, experiment_dir, logger)

    # Training completion
    total_duration = time.time() - experiment_start_time
    
    logger.info("=" * 80)
    logger.info("✅ Training Complete!")
    logger.info(f"🏆 Best Test Accuracy: {best_accuracy:.2f}% achieved at Epoch {best_epoch}")
    logger.info(f"⏱️ Total training time: {total_duration:.2f}s ({total_duration/60:.1f}min)")

    # Save final model
    torch.save(model.state_dict(), model_path)
    logger.info(f"💾 Final model saved to: {get_relative_path(model_path)}")

    # Save best model path for reference
    logger.info(f"💾 Best model saved to: {get_relative_path(best_model_path)}")
    
    # Save plots to experiment directory
    plot_path = experiment_dir / "training_history.png"
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, save_path=plot_path)
    logger.info(f"📊 Training plots saved to: {get_relative_path(plot_path)}")    
    
    # Prepare final metrics
    final_metrics = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'best_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'total_epochs': epochs,
        'total_duration_seconds': total_duration,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0,
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else 0,
        'final_lr': optimizer.param_groups[0]['lr']
    }
    
    # Save metrics to JSON
    save_metrics_to_json(logger, experiment_dir, final_metrics)
    
    logger.info(f"🎯 Experiment completed successfully!")
    logger.info(f"📁 All results saved in: {get_relative_path(experiment_dir)}")
    
    return final_metrics, experiment_dir

def reset_training_history():
    """Reset training history for new training session"""
    global train_losses, train_accuracies, test_losses, test_accuracies
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []