import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniNet(nn.Module):
    """
    MobileNet as presented in (Howard, et al., 2017)
    """

    def __init__(self):
        super(MiniNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 8, 1, stride=2)
        self.conv5 = nn.Conv2d(8, 2, 1, stride=2)
        self.fc1 = nn.Linear(230, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.batch_norm3(self.conv3(x))), (2, 2))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)
