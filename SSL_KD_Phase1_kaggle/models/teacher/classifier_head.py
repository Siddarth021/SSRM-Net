import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=24, dropout_prob=0.4):
        super(ClassifierHead, self).__init__()
        # Input dim is typically 512 + 10 (age/gender features) = 522
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
