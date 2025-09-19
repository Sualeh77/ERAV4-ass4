import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from config import device, logs_dir

def run_lr_finder(model: nn.Module, 
                  train_loader: DataLoader, 
                  loss_fn: nn.Module,
                  optimizer: type,
                  start_lr: float = 1e-7,
                  end_lr: float = 10,
                  num_iter: int = 500,
                  step_mode: str = 'exp',
                  smooth_f: float = 0.05,
                  save_path: Optional[Path] = None,
                  logger = None) -> Tuple[float, float]:
    """
    Run LR Finder to find optimal learning rate
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        loss_fn: Loss function
        optimizer_class: Optimizer class (default: SGD)
        optimizer_kwargs: Additional optimizer parameters
        start_lr: Starting learning rate for search
        end_lr: Ending learning rate for search
        num_iter: Number of iterations for LR search
        step_mode: How to step between start_lr and end_lr ('exp' or 'linear')
        smooth_f: Smoothing factor for loss curve
        save_path: Path to save the LR finder plot
        logger: Logger instance for logging
        
    Returns:
        Tuple of (suggested_lr, steepest_lr)
    """
    if not save_path:
        save_path = Path.cwd() / "lr_finder.png"
    
    # Initialize LR Finder
    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    
    if logger:
        logger.info("🔍 Starting Learning Rate Finder...")
        logger.info(f"   Search range: {start_lr:.2e} to {end_lr:.2e}")
        logger.info(f"   Number of iterations: {num_iter}")
        logger.info(f"   Step mode: {step_mode}")
    
    # Run the LR range test
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        step_mode=step_mode,
        smooth_f=smooth_f
    )
    
    # Get suggested learning rate using the plot method
    # The plot method returns (ax, suggested_lr)
    if save_path:
        plt.figure(figsize=(10, 6))
        ax, suggested_lr = lr_finder.plot(skip_start=10, skip_end=5, suggest_lr=True, ax=plt.gca())
        plt.title('Learning Rate Finder', fontsize=16, fontweight='bold')
        plt.xlabel('Learning Rate', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotation for suggested learning rate
        plt.axvline(x=suggested_lr, color='red', linestyle='--', alpha=0.7, 
                   label=f'Suggested LR: {suggested_lr:.2e}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if logger:
            logger.info(f"📊 LR Finder plot saved to: {save_path}")
    else:
        # If no save path, just get the suggested LR without plotting
        _, suggested_lr = lr_finder.plot(skip_start=10, skip_end=5, suggest_lr=True)
    
    if logger:
        logger.info(f"📊 LR Finder Results:")
        logger.info(f"   Suggested LR: {suggested_lr:.2e}")
    
    # Reset the model and optimizer to original state
    lr_finder.reset()
    
    return suggested_lr

def create_onecycle_scheduler(optimizer: torch.optim.Optimizer,
                             max_lr: float,
                             epochs: int,
                             steps_per_epoch: int,
                             pct_start: float = 0.3,
                             anneal_strategy: str = 'linear',
                             cycle_momentum: bool = True,
                             base_momentum: float = 0.85,
                             max_momentum: float = 0.95,
                             div_factor: float = 100.0,
                             final_div_factor: float = 10000.0,
                             logger = None) -> torch.optim.lr_scheduler.OneCycleLR:
    """
    Create OneCycleLR scheduler with recommended settings
    
    Args:
        optimizer: The optimizer to schedule
        max_lr: Maximum learning rate (from LR Finder)
        epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        pct_start: Percentage of cycle spent increasing LR
        anneal_strategy: 'cos' or 'linear' annealing
        cycle_momentum: Whether to cycle momentum
        base_momentum: Lower momentum bound
        max_momentum: Upper momentum bound
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = initial_lr / final_div_factor
        logger: Logger instance
        
    Returns:
        OneCycleLR scheduler
    """
    
    total_steps = epochs * steps_per_epoch
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        base_momentum=base_momentum,
        max_momentum=max_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )
    
    if logger:
        logger.info(f"📈 OneCycleLR Scheduler Configuration:")
        logger.info(f"   Max LR: {max_lr:.2e}")
        logger.info(f"   Initial LR: {max_lr/div_factor:.2e}")
        logger.info(f"   Final LR: {max_lr/div_factor/final_div_factor:.2e}")
        logger.info(f"   Total steps: {total_steps}")
        logger.info(f"   Pct start: {pct_start}")
        logger.info(f"   Anneal strategy: {anneal_strategy}")
        logger.info(f"   Cycle momentum: {cycle_momentum}")
    
    return scheduler

def suggest_onecycle_max_lr(suggested_lr: float, 
                           strategy: str = 'conservative') -> float:
    """
    Suggest max_lr for OneCycleLR based on LR Finder results
    
    Args:
        suggested_lr: LR suggested by LR Finder
        steepest_lr: LR at steepest point
        strategy: 'conservative', 'manual'
        
    Returns:
        Recommended max_lr for OneCycleLR
    """
    
    if strategy == 'conservative':
        # Use suggested LR directly (safest)
        max_lr = suggested_lr
    elif strategy == 'manual':
        max_lr = float(input("Please enter the max lr to be used for Onecycle policy"))
    else:
        raise ValueError("Strategy must be 'conservative', 'moderate', or 'aggressive'")
    
    return max_lr

def setup_onecycle_policy(epochs,lr_finder_kwargs, onecycle_kwargs, train_loader,
                          experiment_dir, model, loss_fn, optimizer, logger, find_lr=False):
     # Step 1: Run LR Finder
    logger.info("🔍 Running LR Finder to determine optimal learning rate...")

    # Suggest max_lr for OneCycleLR
    strategy = onecycle_kwargs.get('lr_strategy', 'manual')
    if find_lr:
        # Default LR Finder parameters
        default_lr_finder_kwargs = {
            'start_lr': 1e-7,
            'end_lr': 10,
            'num_iter': min(100, len(train_loader)),
            'step_mode': 'exp',
            'smooth_f': 0.05
        }
        default_lr_finder_kwargs.update(lr_finder_kwargs)

        # Save LR Finder plot
        lr_finder_plot_path = experiment_dir / "lr_finder.png"

        suggested_lr = run_lr_finder(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            save_path=lr_finder_plot_path,
            logger=logger,
            **default_lr_finder_kwargs
        )
        max_lr = suggest_onecycle_max_lr(suggested_lr, strategy)
    else:
        max_lr = float(input("Please enter the max lr to be used for Onecycle policy"))
    
    logger.info(f"📈 Using max_lr = {max_lr:.2e} for OneCycleLR (strategy: {strategy})")
    
    # Step 2: Update optimizer with initial LR
    initial_lr = max_lr / onecycle_kwargs.get('div_factor', 100.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr

     # Step 3: Create OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    
    default_onecycle_kwargs = {
        'pct_start': 0.3,
        'anneal_strategy': 'linear',
        'cycle_momentum': True,
        'base_momentum': 0.85,
        'max_momentum': 0.95,
        'div_factor': 100.0,
        'final_div_factor': 10000.0
    }
    default_onecycle_kwargs.update(onecycle_kwargs)
    
    scheduler = create_onecycle_scheduler(
        optimizer=optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        logger=logger,
        **{k: v for k, v in default_onecycle_kwargs.items() if k != 'lr_strategy'}
    )

    return scheduler


