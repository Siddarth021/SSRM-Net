import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, feature_dim=512, dropout_prob=0.2):
        super(FusionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, rhythm_feats, morph_feats):
        # rhythm_feats: (B, 256), morph_feats: (B, 256)
        x = torch.cat((rhythm_feats, morph_feats), dim=1)  # (B, 512)
        out = self.mlp(x)  # (B, 256)
        return out
