import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.ssl.contrastive_loss import InfoNCELoss

def test_infonce_loss():
    loss_fn = InfoNCELoss(temperature=0.07)
    feats_a = torch.randn(4, 128)
    feats_b = torch.randn(4, 128)
    loss = loss_fn(feats_a, feats_b)
    assert loss.item() > 0
    print("InfoNCE loss pass test passed!")
