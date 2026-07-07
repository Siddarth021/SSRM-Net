import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.training.finetune_teacher import TeacherFineTuneModel

def test_finetune_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inputs: B=8
    lead2 = torch.randn(8, 1, 5000).to(device)
    morphology = torch.randn(8, 11, 1250).to(device)
    y = torch.randint(0, 2, (8, 5)).float().to(device) # B=8, 5 disease classes
    
    criterion = nn.BCEWithLogitsLoss()
    
    # ------------------ Mode 1: Frozen Mode ------------------
    model_frozen = TeacherFineTuneModel(num_classes=5, mode='frozen').to(device)
    
    logits_frozen = model_frozen(lead2, morphology)
    assert logits_frozen.shape == (8, 5), f"Expected shape (8, 5), got {logits_frozen.shape}"
    
    loss_frozen = criterion(logits_frozen, y)
    loss_frozen.backward()
    
    # Check that gradients exist only for classifier
    for name, param in model_frozen.named_parameters():
        if "encoder" in name:
            assert param.grad is None or torch.all(param.grad == 0), f"Encoder param {name} should have no gradients in frozen mode."
        elif "classifier" in name:
            assert param.grad is not None, f"Classifier param {name} should have gradients."
            
    assert not torch.isnan(loss_frozen), "Frozen mode loss is NaN"
    print("Frozen mode passed")
    
    # ------------------ Mode 2: Fine-Tuned Mode ------------------
    model_finetune = TeacherFineTuneModel(num_classes=5, mode='finetune').to(device)
    
    logits_finetune = model_finetune(lead2, morphology)
    assert logits_finetune.shape == (8, 5), f"Expected shape (8, 5), got {logits_finetune.shape}"
    
    loss_finetune = criterion(logits_finetune, y)
    loss_finetune.backward()
    
    # Check that gradients exist for both encoder and classifier
    for name, param in model_finetune.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have gradients in finetune mode."
        
    assert not torch.isnan(loss_finetune), "Finetune mode loss is NaN"
    print("Finetune mode passed")
    
    # ------------------ Parameter Reporting ------------------
    teacher_params = sum(p.numel() for p in model_frozen.encoder.parameters())
    classifier_params = sum(p.numel() for p in model_frozen.classifier.parameters())
    total_params = teacher_params + classifier_params
    
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total parameters: {total_params:,}")

if __name__ == "__main__":
    test_finetune_pipeline()
