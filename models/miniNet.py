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
        self.conv4 = nn.Conv2d(32, 8, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.batch_norm3(self.conv3(x))), (2, 2))
        x = self.conv4(x)
        return torch.sigmoid(x)
