import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5, beta=0.3, gamma=0.2):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, student_projected, teacher_logits, teacher_embedding, labels):
        # 1. Supervised BCE Loss
        bce = self.bce_loss(student_logits, labels)
        
        # 2. Feature Matching MSE Loss
        feature_mse = self.mse_loss(student_projected, teacher_embedding)
        
        # 3. KL Divergence Loss for multi-label logits (Sigmoid-based binary distributions)
        # Scale by temperature
        s_prob = torch.sigmoid(student_logits / self.temperature)
        t_prob = torch.sigmoid(teacher_logits / self.temperature)
        
        # Construct binary distribution [p, 1-p] for each of the classes
        s_dist = torch.stack([s_prob, 1.0 - s_prob], dim=-1)
        t_dist = torch.stack([t_prob, 1.0 - t_prob], dim=-1)
        
        # log_softmax/log is required for input to F.kl_div
        s_log_dist = torch.log(s_dist + 1e-9)
        
        # Calculate KL Divergence: reduction='batchmean' and scale by T^2
        kl_div = F.kl_div(s_log_dist, t_dist, reduction='batchmean') * (self.temperature ** 2)
        
        # Total Weighted Loss
        total_loss = (self.alpha * bce) + (self.beta * feature_mse) + (self.gamma * kl_div)
        
        return {
            "loss": total_loss,
            "bce": bce,
            "feature_mse": feature_mse,
            "kl_div": kl_div
        }
