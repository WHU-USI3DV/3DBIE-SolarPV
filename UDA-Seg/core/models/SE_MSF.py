import torch.nn.functional as F
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # Squeeze
        y = self.fc(y).view(b, c, 1, 1)  # Excitation
        return y.expand_as(x)


class ConvDown(nn.Module):
    def __init__(self):
        super(ConvDown, self).__init__()

        self.new_conv = nn.Conv2d(3072, 2048, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.new_bn = nn.BatchNorm2d(2048)
        # initialization
        # conv_tensor = torch.tensor(np.full([2048, 3072, 1, 1], 0, dtype=float)).float()
        # for i in range(0, 2048):
        #     conv_tensor[i, i, 0, 0] = 1
        # self.new_conv.weight = torch.nn.Parameter(conv_tensor)
        # self.new_conv.bias.data.fill_(0)
        #
        # self.new_bn.weight.data.fill_(1)
        # self.new_bn.bias.data.fill_(0)

    def forward(self, x):
        x = self.new_bn(self.new_conv(x))
        return x