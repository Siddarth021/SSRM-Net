import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

class SSLTrainer(nn.Module):
    def __init__(self, teacher_model, projection_head, augmentations, loss_fn, optimizer, device, scaler=None):
        super(SSLTrainer, self).__init__()
        self.teacher_model = teacher_model.to(device)
        self.projection_head = projection_head.to(device)
        self.augmentations = augmentations
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler

    def train_step(self, lead2, morphology):
        # lead2: (B, 1, 4096), morphology: (B, 11, 4096)
        lead2 = lead2.to(self.device)
        morphology = morphology.to(self.device)
        
        # Apply augmentations twice independently
        lead2_aug_a = self.augmentations(lead2)
        lead2_aug_b = self.augmentations(lead2)
        
        morphology_aug_a = self.augmentations(morphology)
        morphology_aug_b = self.augmentations(morphology)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs1 = self.teacher_model(lead2_aug_a, morphology_aug_a)
                outputs2 = self.teacher_model(lead2_aug_b, morphology_aug_b)
                emb1 = outputs1["embedding"]
                emb2 = outputs2["embedding"]
                z1 = self.projection_head(emb1)
                z2 = self.projection_head(emb2)
                loss = self.loss_fn(z1, z2)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision forward pass
            outputs1 = self.teacher_model(lead2_aug_a, morphology_aug_a)
            outputs2 = self.teacher_model(lead2_aug_b, morphology_aug_b)
            emb1 = outputs1["embedding"]
            emb2 = outputs2["embedding"]
            z1 = self.projection_head(emb1)
            z2 = self.projection_head(emb2)
            loss = self.loss_fn(z1, z2)
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
