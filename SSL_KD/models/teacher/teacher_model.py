import torch
import torch.nn as nn
from .rhythm_encoder import RhythmEncoder
from .morphology_encoder import MorphologyEncoder
from .fusion import FusionModule

class TeacherModel(nn.Module):
    def __init__(self, rhythm_dim=256, morph_dim=256, fused_dim=256):
        super(TeacherModel, self).__init__()
        
        self.rhythm_encoder = RhythmEncoder(in_channels=1, out_channels=rhythm_dim)
        self.morphology_encoder = MorphologyEncoder(in_channels=11, out_channels=morph_dim)
        self.fusion = FusionModule(feature_dim=fused_dim)
        
        # MaxPool downsamples 5000 length to 1250 (factor of 4)
        self.downsample = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, lead2, morphology, age_gender=None):
        # Input lead2: (B, 1, 5000)
        # Input morphology: (B, 11, 1250)
        
        # Extract features
        rhythm_feature = self.rhythm_encoder(lead2)  # (B, 256)
        morphology_feature = self.morphology_encoder(morphology)  # (B, 256)
        
        # Fuse features
        embedding = self.fusion(rhythm_feature, morphology_feature)  # (B, 256)
        
        return {
            "embedding": embedding,
            "rhythm_feature": rhythm_feature,
            "morphology_feature": morphology_feature
        }
