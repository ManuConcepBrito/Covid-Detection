import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniNet(nn.Module):

    def __init__(self):
        super(MiniNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2)
        self.relu5 = nn.ReLU()
        self.fc1 = nn.Linear(3584, 128)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(128, 20)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.max_pool2d(self.relu1(self.batch_norm1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(self.relu2(self.batch_norm2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(self.relu3(self.batch_norm3(self.conv3(x))), (2, 2))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = x.view(-1)
        x = self.relu6(self.fc1(x))
        x = self.relu7(self.fc2(x))
        x = self.fc3(x)
        return x
