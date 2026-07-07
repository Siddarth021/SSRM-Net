import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.models.student.student_model import CustomResnet

def test_student_forward():
    model = CustomResnet(input_channel=6, num_classes=24)
    # Batch size 2, 6 leads (lightweight student), 4096 time steps
    x = torch.randn(2, 6, 4096)
    ag = torch.randn(2, 5) # Age gender features
    out = model(x, ag)
    assert out.shape == (2, 24)
    print("Student forward pass test passed!")
