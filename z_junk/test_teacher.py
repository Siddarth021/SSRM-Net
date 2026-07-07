import torch
import sys
import os

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.models.teacher.teacher_model import resnet18

def test_teacher_forward():
    model = resnet18(in_channel=12, out_channel=24)
    # Batch size 2, 12 leads, 4096 time steps
    x = torch.randn(2, 12, 4096)
    ag = torch.randn(2, 5) # Age gender features
    out = model(x, ag)
    assert out.shape == (2, 24)
    print("Teacher forward pass test passed!")
