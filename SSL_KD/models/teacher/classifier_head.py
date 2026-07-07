import torch
import torch.nn as nn

class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=5, dropout_prob=0.3):
        super(ClassifierHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
