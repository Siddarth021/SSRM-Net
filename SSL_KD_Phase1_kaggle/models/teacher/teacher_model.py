import torch
import torch.nn as nn
from .rhythm_encoder import RhythmEncoder
from .morphology_encoder import MorphologyEncoder
from .fusion import FusionModule

class TeacherModel(nn.Module):
    def __init__(self, rhythm_dim=512, morph_dim=512, fused_dim=512):
        super(TeacherModel, self).__init__()
        
        self.rhythm_encoder = RhythmEncoder(in_channels=1, out_channels=rhythm_dim)
        self.morphology_encoder = MorphologyEncoder(in_channels=11, out_channels=morph_dim)
        self.fusion = FusionModule(feature_dim=fused_dim)
        
        # Age/Gender processing
        self.age_gender_fc = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(inplace=True)
        )
        
        # MaxPool downsamples 5000 length to 1250 (factor of 4)
        self.downsample = nn.MaxPool1d(kernel_size=4, stride=4)

    def forward(self, lead2, morphology, age_gender=None):
        # Input lead2: (B, 1, 5000)
        # Input morphology: (B, 11, 1250)
        
        # Extract features
        rhythm_feature = self.rhythm_encoder(lead2)  # (B, 256)
        morphology_feature = self.morphology_encoder(morphology)  # (B, 256)
        
        # Fuse features
        embedding = self.fusion(rhythm_feature, morphology_feature)  # (B, 512)
        
        # Add age/gender if provided
        if age_gender is not None:
            ag_feat = self.age_gender_fc(age_gender)
            embedding = torch.cat((ag_feat, embedding), dim=1) # (B, 522)
            
        return {
            "embedding": embedding,
            "rhythm_feature": rhythm_feature,
            "morphology_feature": morphology_feature
        }
