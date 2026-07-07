import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
from .transforms import (
    Compose, Filtering, ZScore, NaNvalues, Normalize, RandomClip, Retype
)

# Standard CINC 2020 Equivalent Classes (27 scored classes merged into 24)
EQUIVALENT_CLASSES = {
    "713427006": "59118001",  # CRBBB -> RBBB
    "284470004": "63593006",  # PAC -> SVPB
    "427172004": "17338001"   # PVC -> VPB
}

def load_cinc_header(header_file):
    """Parses a PhysioNet .hea file to extract SNOMED CT codes and sampling frequency."""
    with open(header_file, 'r') as f:
        lines = f.readlines()
    
    # Extract sampling frequency from the first line
    # Format: RecordName NumLeads Fs NumSamples
    header_info = lines[0].strip().split()
    fs = int(header_info[2])
    
    # Extract diagnoses
    diagnoses = []
    for line in lines:
        if '#Dx:' in line.replace(' ', ''):
            # Format: # Dx: 164889003,426783006
            dx_str = line.split('Dx:')[1].strip()
            codes = dx_str.split(',')
            for code in codes:
                code = code.strip()
                if code:
                    diagnoses.append(code)
            break
            
    return fs, diagnoses

def load_cinc_signal(mat_file):
    """Loads the .mat signal file."""
    x = loadmat(mat_file)
    signal = np.asarray(x['val'], dtype=np.float64)
    return signal

class CINC2020Dataset(Dataset):
    def __init__(self, data_dir, split_df, test=False, transform=None, target_fs=257, seq_length=4096, lead2_sec=15.9, morphology_sec=2.5):
        """
        data_dir: Path to the dataset
        split_df: Pandas dataframe containing 'filename' column (path to record without extension)
        """
        self.data_dir = data_dir
        self.test = test
        self.transform = transform
        self.target_fs = target_fs
        self.seq_length = seq_length
        self.lead2_sec = lead2_sec
        self.morphology_sec = morphology_sec
        self.data = split_df['filename'].tolist()
        
        # Load scored classes mapping
        mapping_path = os.path.join(data_dir, 'dx_mapping_scored.csv')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Missing {mapping_path}. Please ensure CINC 2020 dataset is properly formatted.")
            
        dx_df = pd.read_csv(mapping_path)
        self.scored_classes = dx_df['SNOMED CT Code'].astype(str).tolist()
        
        # Apply equivalencies to find the unique 24 classes
        mapped_classes = set()
        for code in self.scored_classes:
            mapped_classes.add(EQUIVALENT_CLASSES.get(code, code))
        
        self.target_classes = sorted(list(mapped_classes))
        self.num_classes = len(self.target_classes)
        
        if self.num_classes != 24:
            print(f"Warning: Expected 24 classes after merging, but got {self.num_classes}")
            
        self.class_to_idx = {str(code): i for i, code in enumerate(self.target_classes)}

    def __len__(self):
        return len(self.data)

    def _get_multi_hot_label(self, diagnoses):
        label = np.zeros(self.num_classes, dtype=np.float32)
        for dx in diagnoses:
            # Apply equivalency mapping
            mapped_dx = EQUIVALENT_CLASSES.get(dx, dx)
            if mapped_dx in self.class_to_idx:
                label[self.class_to_idx[mapped_dx]] = 1.0
        return label

    def __getitem__(self, item):
        record_path = os.path.join(self.data_dir, self.data[item])
        if record_path.endswith('.mat'):
            record_path = record_path[:-4]
        
        hea_file = record_path + '.hea'
        mat_file = record_path + '.mat'
        
        # Load Header
        fs, diagnoses = load_cinc_header(hea_file)
        
        # Load Signal
        signal = load_cinc_signal(mat_file)
        
        # Resample to target_fs if necessary
        if fs != self.target_fs:
            num_samples = int(signal.shape[1] * self.target_fs / fs)
            signal = resample(signal, num_samples, axis=1)
            
        if self.transform:
            signal = self.transform(signal)

        # Calculate exact number of samples for each branch
        lead2_samples = min(int(self.lead2_sec * self.target_fs), self.seq_length)
        morph_samples = min(int(self.morphology_sec * self.target_fs), self.seq_length)

        # Split leads
        # Standard CINC 2020 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        # Lead II is at index 1
        lead2 = signal[1:2, :lead2_samples]
        
        # Remaining leads
        remaining_indices = [0] + list(range(2, 12))
        morphology = signal[remaining_indices, :morph_samples]

        # Convert to PyTorch tensors
        if not isinstance(lead2, torch.Tensor):
            lead2_tensor = torch.from_numpy(lead2).float()
        else:
            lead2_tensor = lead2.float()

        if not isinstance(morphology, torch.Tensor):
            morphology_tensor = torch.from_numpy(morphology).float()
        else:
            morphology_tensor = morphology.float()

        if self.test:
            return {
                "lead2": lead2_tensor,
                "morphology": morphology_tensor,
                "filename": self.data[item]
            }
        else:
            label = self._get_multi_hot_label(diagnoses)
            label_tensor = torch.from_numpy(label).float()
            return {
                "lead2": lead2_tensor,
                "morphology": morphology_tensor,
                "label": label_tensor,
                "filename": self.data[item]
            }

class CINC2020ECG(object):
    def __init__(self, data_dir, split='0', target_fs=257, seq_length=4096, lead2_sec=15.9, morphology_sec=2.5):
        self.data_dir = data_dir
        self.split = split
        self.target_fs = target_fs
        self.seq_length = seq_length
        self.lead2_sec = lead2_sec
        self.morphology_sec = morphology_sec
        
        # We assume 24 classes for CINC 2020
        self.num_classes = 24

        normlizetype = 'mean-std'
        
        self.data_transforms = {
            'train': Compose([
                Filtering(),
                ZScore(),
                NaNvalues(),
                Normalize(normlizetype),
                RandomClip(len=self.seq_length),
                Retype()
            ]),
            'val': Compose([
                Filtering(),
                ZScore(),
                NaNvalues(),
                Normalize(normlizetype),
                RandomClip(len=self.seq_length),
                Retype()
            ]),
            'test': Compose([
                Filtering(),
                ZScore(),
                NaNvalues(),
                Normalize(normlizetype),
                RandomClip(len=self.seq_length),
                Retype()
            ])
        }

    def data_prepare(self, test=False):
        # Use the 5-fold cross validation splits from the '5folds' directory
        folds_dir = os.path.join(self.data_dir, '5folds')
        
        train_path = os.path.join(folds_dir, f'train_split{self.split}.csv')
        test_path = os.path.join(folds_dir, f'test_split{self.split}.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing fold files for split {self.split}. Expected {train_path} and {test_path}")
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # LRH_Net uses 'test_splitX.csv' for validation. We will use it for both val and test datasets here.
        train_dataset = CINC2020Dataset(
            self.data_dir, train_df, transform=self.data_transforms['train'], 
            target_fs=self.target_fs, seq_length=self.seq_length,
            lead2_sec=self.lead2_sec, morphology_sec=self.morphology_sec
        )
        val_dataset = CINC2020Dataset(
            self.data_dir, test_df, transform=self.data_transforms['val'], 
            target_fs=self.target_fs, seq_length=self.seq_length,
            lead2_sec=self.lead2_sec, morphology_sec=self.morphology_sec
        )
        test_dataset = CINC2020Dataset(
            self.data_dir, test_df, transform=self.data_transforms['test'], 
            target_fs=self.target_fs, seq_length=self.seq_length,
            lead2_sec=self.lead2_sec, morphology_sec=self.morphology_sec
        )
        
        return train_dataset, val_dataset, test_dataset
