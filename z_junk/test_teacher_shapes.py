import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.models.teacher.teacher_model import TeacherModel

def test_teacher_shapes():
    # B=4
    lead2 = torch.randn(4, 1, 5000)
    morphology = torch.randn(4, 11, 1250)
    model = TeacherModel()
    
    # Run forward pass
    outputs = model(lead2, morphology)
    
    # Verify shapes
    embedding = outputs["embedding"]
    rhythm_feature = outputs["rhythm_feature"]
    morphology_feature = outputs["morphology_feature"]
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Rhythm feature shape: {rhythm_feature.shape}")
    print(f"Morphology feature shape: {morphology_feature.shape}")
    
    assert embedding.shape == (4, 256), f"Expected embedding shape (4, 256), got {embedding.shape}"
    assert rhythm_feature.shape == (4, 256), f"Expected rhythm_feature shape (4, 256), got {rhythm_feature.shape}"
    assert morphology_feature.shape == (4, 256), f"Expected morphology_feature shape (4, 256), got {morphology_feature.shape}"
    
    print("Shape assertions passed successfully!")
    
    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    test_teacher_shapes()
