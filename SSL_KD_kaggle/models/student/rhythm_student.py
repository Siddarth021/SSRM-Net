import torch
import torch.nn as nn
import torch.nn.functional as F

class RhythmStudentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RhythmStudentResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class RhythmStudentEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(RhythmStudentEncoder, self).__init__()
        # Input shape: (B, 1, 5000)
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=False)
        ) # Output: (B, 16, 2500)
        
        self.res_block1 = RhythmStudentResidualBlock(16, 16, stride=1) # (B, 16, 2500)
        self.res_block2 = RhythmStudentResidualBlock(16, 32, stride=2) # (B, 32, 1250)
        
        # BiGRU: input_size=32, hidden_size=32, bidirectional=True (output size: 64)
        self.gru = nn.GRU(input_size=32, hidden_size=32, num_layers=1,
                          batch_first=True, bidirectional=True)
        
        # Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False)
        )

    def forward(self, x):
        # x: (B, 1, 5000)
        out = self.init_conv(x)  # (B, 16, 2500)
        out = self.res_block1(out) # (B, 16, 2500)
        out = self.res_block2(out) # (B, 32, 1250)
        
        # GRU expects shape: (B, L, C)
        out = out.transpose(1, 2) # (B, 1250, 32)
        gru_out, _ = self.gru(out) # (B, 1250, 64)
        
        # Attention pooling
        attn_weights = self.attention(gru_out) # (B, 1250, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        rhythm_feat = torch.sum(gru_out * attn_weights, dim=1) # (B, 64)
        return rhythm_feat
