import torch
import torch.nn as nn
import torch.nn.functional as F

class RhythmResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RhythmResidualBlock, self).__init__()
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


class RhythmEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super(RhythmEncoder, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False)
        )
        
        self.res_blocks = nn.Sequential(
            RhythmResidualBlock(64, 64, stride=1),
            RhythmResidualBlock(64, 128, stride=2),
            RhythmResidualBlock(128, 256, stride=2),
            RhythmResidualBlock(256, 256, stride=1)
        )
        
        # BiLSTM input_size is 256 (channels from last res block), hidden_size is 256.
        # Since it is bidirectional, the output size is 256 * 2 = 512.
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, 
                            batch_first=True, bidirectional=True)
        
        # Temporal Attention
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

    def forward(self, x):
        # x: (B, 1, 5000)
        out = self.init_conv(x)  # (B, 64, 2500)
        out = self.res_blocks(out)  # (B, 128, 1250)
        
        # Prepare for LSTM: (B, L, C)
        out = out.transpose(1, 2)  # (B, 1250, 128)
        # Force LSTM to run in float32 to avoid cuDNN mixed precision issues on some GPUs
        with torch.amp.autocast('cuda', enabled=False):
            lstm_out, _ = self.lstm(out.float())
        # Cast back to the current default dtype (FP16 if in autocast)
        lstm_out = lstm_out.to(x.dtype)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, 1250, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        rhythm_feat = torch.sum(lstm_out * attn_weights, dim=1)  # (B, 256)
        return rhythm_feat
