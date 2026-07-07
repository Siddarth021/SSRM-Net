import torch
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.models.teacher.teacher_model import TeacherModel
from ECG_SSL_KD.ssl.projection_head import ProjectionHead
from ECG_SSL_KD.ssl.augmentations import ComposeSSL, RandomMask, GaussianNoise, AmplitudeScaling, TimeShift, LeadDropout
from ECG_SSL_KD.ssl.contrastive_loss import NTXentLoss
from ECG_SSL_KD.ssl.ssl_trainer import SSLTrainer

def test_ssl_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize components
    teacher = TeacherModel().to(device)
    projection_head = ProjectionHead().to(device)
    
    # Compose SSL Augmentations
    augmentations = ComposeSSL([
        RandomMask(0.1, 0.3),
        GaussianNoise(0.02),
        AmplitudeScaling(0.8, 1.2),
        TimeShift(0.05),
        LeadDropout(0.3)
    ])
    
    loss_fn = NTXentLoss(temperature=0.1)
    
    # Optimizing both teacher and projection head
    params = list(teacher.parameters()) + list(projection_head.parameters())
    optimizer = optim.Adam(params, lr=0.0003)
    
    trainer = SSLTrainer(teacher, projection_head, augmentations, loss_fn, optimizer, device)
    
    # Input batch: B=8
    lead2 = torch.randn(8, 1, 5000).to(device)
    morphology = torch.randn(8, 11, 1250).to(device)
    
    # Direct flow checks for output shapes
    with torch.no_grad():
        aug_lead2 = augmentations(lead2)
        aug_morph = augmentations(morphology)
        teacher_out = teacher(aug_lead2, aug_morph)
        emb = teacher_out["embedding"]
        proj = projection_head(emb)
    
    print(f"Teacher embedding shape: {emb.shape}")
    print(f"Projection shape: {proj.shape}")
    
    assert emb.shape == (8, 256), f"Expected embedding (8, 256), got {emb.shape}"
    assert proj.shape == (8, 128), f"Expected projection (8, 128), got {proj.shape}"
    
    # Run one trainer step
    loss_val = trainer.train_step(lead2, morphology)
    print(f"Loss value: {loss_val}")
    
    # Check if loss is a valid finite scalar
    assert isinstance(loss_val, float), "Expected loss to be a float"
    assert not torch.isnan(torch.tensor(loss_val)), "Loss is NaN"
    
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Trainable parameter count: {trainable_params:,}")
    print("SSL pipeline passed.")

if __name__ == "__main__":
    test_ssl_pipeline()
