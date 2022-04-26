import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 1200),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 1200),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1200, num_classes),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.features(x)

        return x


# 写法二
# class TeacherModel(nn.Module):
#     def __init__(self, in_channels=1, num_classes=10):
#         super(TeacherModel, self).__init__()
    #     self.relu = nn.ReLU()
    #     self.fc1 = nn.Linear(784, 1200)
    #     self.fc2 = nn.Linear(1200, 1200)
    #     self.fc3 = nn.Linear(1200, num_classes)
    #     self.dropout = nn.Dropout(p=0.5)
    #
    # def forward(self, x):
    #     x = x.view(-1, 784)
    #     x = self.fc1(x)
    #     x = self.dropout(x)
    #     x = self.relu(x)
    #
    #     x = self.fc2(x)
    #     x = self.dropout(x)
    #     x = self.relu(x)
    #
    #     x = self.fc3(x)
    #
    #     return x