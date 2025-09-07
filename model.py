from torch import nn

class MnistFullyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            # nn.Dropout(p=0.05),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1), # 28x28 -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.05),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        )
        
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(p=0.05),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.05),
            nn.Conv2d(32, 16, kernel_size=1, stride=1)  # 1x1 conv for dimension reduction
        )
        
        # Final classification layer - fully convolutional
        self.block4 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=1, stride=1),  # 1x1 conv for 10 classes
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 10)
        return x