# ml/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessValueNet(nn.Module):
    """
    Simple convolutional value network.

    Input: [B, 18, 8, 8]
    Output: [B] scalar in [-1,1] (via tanh)
    """

    def __init__(self, in_channels: int = 18, hidden_channels: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(hidden_channels * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: [B, C, 8, 8]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # [-1,1]
        return x.squeeze(-1)         # [B]
