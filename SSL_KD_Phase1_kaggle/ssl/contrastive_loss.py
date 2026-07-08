import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # z1, z2 shapes: (B, 128)
        batch_size = z1.size(0)
        
        # Normalize representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate projections
        representations = torch.cat([z1, z2], dim=0)  # (2B, 128)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)  # (2B, 2B)
        
        # Rescale similarity scores by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Positive keys are (z1_i, z2_i)
        # Create mask for self-similarities and populate target label
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e4)
        
        # Target index labels: representation i's positive partner is at:
        # i + batch_size if i < batch_size else i - batch_size
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
