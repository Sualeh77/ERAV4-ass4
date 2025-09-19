from torchsummary import summary
from config import device, logs_dir, input_size, PROJECT_ROOT
import logging
import json
import time
from datetime import datetime
import os
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
import torch
import io
import sys
from contextlib import redirect_stdout

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def show_model_summary(model, input_size):
  summary(model, input_size, device=device)

# Global experiment start time
experiment_start_time = None

def get_relative_path(path: Path) -> str:
    """Convert absolute path to relative path from PROJECT_ROOT"""
    try:
        return str(Path(path).relative_to(PROJECT_ROOT))
    except ValueError:
        # If path is not relative to PROJECT_ROOT, just return the name
        return str(Path(path).name)

def setup_logging(experiment_name: str = None, log_level: str = "INFO"):
    """
    Setup comprehensive logging for training experiments
    Returns: (experiment_dir, logger)
    """
    global experiment_start_time
    
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(exist_ok=True)
    
    # Generate experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"mnist_experiment_{timestamp}"
    
    experiment_start_time = time.time()
    
    # Create experiment-specific subdirectory
    experiment_dir = logs_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    # Setup logger
    logger = logging.getLogger('mnist_training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs
    log_file = experiment_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log experiment start
    logger.info("=" * 80)
    logger.info(f"🚀 Starting experiment: {experiment_name}")
    logger.info(f"📁 Experiment directory: {get_relative_path(experiment_dir)}")
    logger.info(f"💾 Log file: {get_relative_path(log_file)}")
    logger.info("=" * 80)
    
    return experiment_dir, experiment_start_time, logger

def log_model_info(logger, model: nn.Module):
    """Log model architecture and parameters"""
    if logger is None:
        return
    
    # Capture model summary output
    logger.info("📋 Detailed Model Summary:")
    logger.info("-" * 60)

    try:
        # Create a string buffer to capture the summary output
        # Capture the summary output
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            show_model_summary(model, input_size)
        
        # Get the summary string and log it line by line
        summary_output = summary_buffer.getvalue()
        for line in summary_output.split('\n'):
            if line.strip():  # Only log non-empty lines
                logger.info(f"   {line}")
                
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")
    
    # Log model structure
    logger.debug("Model structure:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            logger.debug(f"   {name}: {module}")

    logger.info("-" * 60)

def log_dataset_info(logger, train_loader: DataLoader, test_loader: DataLoader):
    """Log dataset information"""
    if logger is None:
        return
        
    logger.info("📊 Dataset Information:")
    logger.info(f"   Training batches: {len(train_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    logger.info(f"   Training samples: {len(train_loader.dataset):,}")
    logger.info(f"   Test samples: {len(test_loader.dataset):,}")
    logger.info(f"   Batch size: {train_loader.batch_size}")

def log_training_config(logger, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, scheduler=None):
    """Log training configuration"""
    if logger is None:
        return
        
    logger.info("⚙️ Training Configuration:")
    logger.info(f"   Device: {device}")
    logger.info(f"   Optimizer: {optimizer.__class__.__name__}")
    logger.info(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    logger.info(f"   Loss function: {loss_fn.__class__.__name__}")
    
    if scheduler:
        logger.info(f"   Scheduler: {scheduler.__class__.__name__}")
    
    # Log optimizer parameters
    for group_idx, param_group in enumerate(optimizer.param_groups):
        logger.debug(f"   Optimizer group {group_idx}:")
        for key, value in param_group.items():
            if key != 'params':
                logger.debug(f"     {key}: {value}")

def save_metrics_to_json(logger, experiment_dir: Path, metrics: dict):
    """Save training metrics to JSON file"""
    metrics_file = experiment_dir / "metrics.json"
    
    # Add experiment metadata
    metrics_with_meta = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": time.time() - experiment_start_time if experiment_start_time else 0,
            "device": str(device),
        },
        "metrics": metrics
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_with_meta, f, indent=2)
    
    logger.info(f"💾 Metrics saved to: {get_relative_path(metrics_file)}")