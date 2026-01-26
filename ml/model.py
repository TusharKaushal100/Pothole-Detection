import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # -------- Convolution Block 1 --------
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB
            out_channels=8,     # number of filters
            kernel_size=3,
            padding=1
        )

        # -------- Convolution Block 2 --------
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # -------- Fully Connected Layers --------
        self.fc1 = nn.Linear(16 * 56 * 56, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch, 3, 224, 224]

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)   # 224 → 112

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)   # 112 → 56

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x   # raw score (logit)
