import torch
import torch.nn as nn

class Detector(nn.Module):
    """
    Simple detector that takes classifier logits as input and predicts
    whether the input is clean (0) or adversarial (1)
    """
    def __init__(self, input_dim=10):
        super(Detector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
