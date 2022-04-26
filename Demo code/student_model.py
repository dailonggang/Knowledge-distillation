import torch
import torch.nn as nn


class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, num_classes),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.features(x)

        return x