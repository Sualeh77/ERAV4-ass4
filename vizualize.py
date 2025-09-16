import matplotlib.pyplot as plt

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies, save_path=None):
    """
    Plot training and validation curves
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, test_losses, 'ro-', label='Test Loss', linewidth=2, markersize=6)
    ax1.set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy', linewidth=2, markersize=6)
    ax2.set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()