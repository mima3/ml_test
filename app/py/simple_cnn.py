import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [N,32,26,26]
        x = F.relu(self.conv2(x))   # [N,64,24,24]
        x = F.max_pool2d(x, 2)      # [N,64,12,12]
        x = torch.flatten(x, 1)     # [N,64*12*12]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
