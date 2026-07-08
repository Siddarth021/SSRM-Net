import torch
import torch.nn as nn

class FeatureLoss(nn.Module):
    """
    Loss module for aligning intermediate features between teacher and student.
    """
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        loss = 0.0
        # Align features if they match in shape or apply projection
        for sf, tf in zip(student_features, teacher_features):
            if sf.shape == tf.shape:
                loss += self.mse(sf, tf)
        return loss
