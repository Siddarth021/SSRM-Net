import torch
import torch.nn as nn
from SSL_KD.models.student.rhythm_student import RhythmStudentEncoder
from SSL_KD.models.student.morphology_student import MorphologyStudentEncoder

class StudentModel(nn.Module):
    def __init__(self, num_classes=5):
        super(StudentModel, self).__init__()
        
        self.rhythm_encoder = RhythmStudentEncoder()
        self.morphology_encoder = MorphologyStudentEncoder()
        
        # Fusion MLP: 128 -> 128 -> 64
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )
        
        # Projection layer to match Teacher's 256-d embedding space for feature distillation
        self.projection = nn.Linear(64, 256)
        
        # Classifier Head for downstream classification
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, lead2, morphology):
        # Rhythm branch: (B, 64)
        rhythm_feat = self.rhythm_encoder(lead2)
        # Morphology branch: (B, 64)
        morph_feat = self.morphology_encoder(morphology)
        
        # Concatenate features: (B, 128)
        fused = torch.cat((rhythm_feat, morph_feat), dim=1)
        
        # Final embedding: (B, 64)
        embedding = self.fusion(fused)
        
        # Projected embedding: (B, 256)
        projected = self.projection(embedding)
        
        # Logits: (B, num_classes)
        logits = self.classifier(embedding)
        
        return {
            "embedding": embedding,
            "projected": projected,
            "logits": logits
        }
