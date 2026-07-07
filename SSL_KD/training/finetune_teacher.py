import torch
import torch.nn as nn
from SSL_KD.models.teacher.teacher_model import TeacherModel
from SSL_KD.models.teacher.classifier_head import ClassifierHead

class TeacherFineTuneModel(nn.Module):
    def __init__(self, num_classes=5, mode='frozen'):
        super(TeacherFineTuneModel, self).__init__()
        self.encoder = TeacherModel()
        self.classifier = ClassifierHead(embedding_dim=512, num_classes=num_classes)
        self.set_mode(mode)

    def set_mode(self, mode):
        assert mode in ['frozen', 'finetune'], "Mode must be 'frozen' or 'finetune'"
        self.mode = mode
        if mode == 'frozen':
            # Freeze all encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Ensure classifier parameters are updated
            for param in self.classifier.parameters():
                param.requires_grad = True
        elif mode == 'finetune':
            # Unfreeze everything
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, lead2, morphology):
        # Forward pass through encoder
        outputs = self.encoder(lead2, morphology)
        embedding = outputs["embedding"]
        
        # Forward pass through classifier head
        logits = self.classifier(embedding)
        return logits
