import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, inputs, targets, class_sample_freqn, total_samples):
        class_sample_freqn = torch.FloatTensor(class_sample_freqn).to(inputs.device)
        class_wts = 1.0 / class_sample_freqn
        pos_wts = (total_samples - class_sample_freqn) / class_sample_freqn
        criterion = nn.BCEWithLogitsLoss(weight=class_wts, reduction='none', pos_weight=pos_wts)
        BCE_loss = criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()


class KDLoss(nn.Module):
    def __init__(self, temperature=8.0):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.divergence_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        loss = self.divergence_fn(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        return loss
