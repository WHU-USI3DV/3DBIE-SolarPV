import torch.nn.functional as F
import torch.nn as nn


class DN_transfrom(nn.Module):
    def __init__(self):
        super(DN_transfrom, self).__init__()
        self.new_conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1)
        self.new_bn1 = nn.BatchNorm2d(16)
        self.new_conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)
        self.new_bn2 = nn.BatchNorm2d(32)
        self.new_conv3 = nn.Conv2d(32, 128, kernel_size=1, stride=1)
        self.new_bn3 = nn.BatchNorm2d(128)
        self.new_conv4 = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.new_bn4 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.new_bn1(self.relu(self.new_conv1(x)))
        x = self.new_bn2(self.relu(self.new_conv2(x)))
        x = self.new_bn3(self.relu(self.new_conv3(x)))
        x = self.new_bn4(self.tanh(self.new_conv4(x)))
        return x