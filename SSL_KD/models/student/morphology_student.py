import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentDepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(StudentDepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(self.bn(out))
        return out


class MorphologyStudentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MorphologyStudentResidualBlock, self).__init__()
        self.sep_conv1 = StudentDepthwiseSeparableConv1D(in_channels, out_channels, stride=stride)
        self.sep_conv2 = StudentDepthwiseSeparableConv1D(out_channels, out_channels, stride=1)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = out + self.shortcut(x)
        return out


class MorphologyStudentEncoder(nn.Module):
    def __init__(self, in_channels=11, out_channels=64):
        super(MorphologyStudentEncoder, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False)
        )
        
        self.blocks = nn.Sequential(
            MorphologyStudentResidualBlock(16, 32, stride=2),   # length to 625
            MorphologyStudentResidualBlock(32, 64, stride=2)    # length to 313
        )
        
        # Lead/Channel Attention
        self.lead_attn = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 11, 1250)
        out = self.init_conv(x)  # (B, 16, 1250)
        out = self.blocks(out)   # (B, 64, 313)
        
        # Lead Attention (channel-wise)
        # Average over temporal dimension
        gap_temp = out.mean(dim=-1)  # (B, 64)
        attn_weights = self.lead_attn(gap_temp).unsqueeze(-1)  # (B, 64, 1)
        out = out * attn_weights
        
        # Global Average Pooling
        morph_feat = out.mean(dim=-1)  # (B, 64)
        return morph_feat
