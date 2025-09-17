import torch
import torchmetrics
from typing import Dict

class MetricsTracker:
    """Helper class to track multiple metrics"""
    def __init__(self, num_classes: int = 10, device: str = 'cpu'):
        self.device = device
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
        self.recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update all metrics"""
        self.accuracy.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1.update(preds, targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'accuracy': self.accuracy.compute().item() * 100,
            'precision': self.precision.compute().item() * 100,
            'recall': self.recall.compute().item() * 100,
            'f1_score': self.f1.compute().item() * 100
        }
    
    def reset(self):
        """Reset all metrics"""
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()