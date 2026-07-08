import torch
import torch.nn as nn

def conv1d(in_planes, out_planes, kernel_size, strides=1, padding='same', bias=True):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=strides, padding=padding, bias=bias)


class Cust_SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(Cust_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Cust_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_channels, kernel_size=8, stride=1, padding='same', bias=False, downsample=None):
        super().__init__()
        self.conv1 = conv1d(in_planes, out_channels, kernel_size, strides=(1 if not downsample else 2), padding=('same' if not downsample else 3))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv1d(out_channels, out_channels, kernel_size, strides=1)
        self.se = Cust_SELayer(out_channels)
        self.downsample = downsample
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.bn2(out)    
        return out
