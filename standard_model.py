
import torch
from torch import nn

class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 80, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(80, 96, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

