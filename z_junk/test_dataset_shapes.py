import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ECG_SSL_KD.datasets.ptbxl_dataset import PTBXLDataset

def test_dataset_shapes():
    # Construct a mock anno_pd
    data = {
        'filename': ['dummy1.mat', 'dummy2.mat'],
        'fs': [500, 500],
        'age': [50, 60],
        'gender': ['M', 'F'],
        'NORM': [1, 0],
        'CD': [0, 1],
        'MI': [1, 0],
        'HYP': [0, 0],
        'STTC': [0, 1]
    }
    anno_pd = pd.DataFrame(data)
    
    # Mock data loader returning shape (12, 5000)
    def mock_loader(path, src_fs):
        return np.random.randn(12, 5000)
        
    ds = PTBXLDataset(anno_pd=anno_pd, loader=mock_loader)
    
    sample = ds[0]
    
    lead2 = sample["lead2"]
    morphology = sample["morphology"]
    label = sample["label"]
    
    print(f"Dataset lead2 shape: {lead2.shape}")
    print(f"Dataset morphology shape: {morphology.shape}")
    print(f"Dataset label shape: {label.shape}")
    
    assert lead2.shape == (1, 5000), f"Expected (1, 5000), got {lead2.shape}"
    assert morphology.shape == (11, 1250), f"Expected (11, 1250), got {morphology.shape}"
    assert label.shape == (5,), f"Expected (5,), got {label.shape}"
    
    print("Dataset shapes check passed successfully!")

if __name__ == "__main__":
    test_dataset_shapes()
