"""
Pre-process all CINC2020 .mat/.hea files into a single .pt tensor file.
This eliminates per-sample disk I/O during training, speeding up epochs dramatically.

Usage:
    python preprocess_cinc2020.py --data_dir /kaggle/input/datasets/gundetisiddarth/physionet-2020/phsionet_2020/ --target_fs 257 --seq_length 4096

The output file is saved to: <data_dir>/preprocessed/cinc2020_12lead_257hz_4096len.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import loadmat
from scipy import signal as scipy_signal
from scipy.stats import zscore


def Resample(input_signal, src_fs, tar_fs):
    """Fast linear interpolation resampling (matches LRH-Net)."""
    if src_fs != tar_fs:
        dtype = input_signal.dtype
        audio_len = input_signal.shape[1]
        audio_time_max = 1.0 * audio_len / src_fs
        src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0, int(audio_time_max * tar_fs), int(audio_time_max * tar_fs)) / tar_fs
        output_channels = []
        for i in range(input_signal.shape[0]):
            resampled = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
            output_channels.append(resampled.reshape(1, -1))
        output_signal = np.vstack(output_channels)
    else:
        output_signal = input_signal
    return output_signal


def apply_fixed_transforms(signal, seq_length):
    """Apply deterministic transforms: Filtering -> ZScore -> NaN removal -> Normalize -> Pad/Clip."""
    # 1. Bandpass filter (0.001-47 Hz at 250 Hz normalized)
    b, a = scipy_signal.butter(3, [0.001 / 250, 47 / 250], 'bandpass')
    signal = scipy_signal.filtfilt(b, a, signal)
    
    # 2. Z-score normalization
    signal = zscore(signal, axis=-1)
    
    # 3. Replace NaN values
    signal = np.nan_to_num(signal)
    
    # 4. Mean-std normalization per channel
    for i in range(signal.shape[0]):
        if np.sum(signal[i, :]) != 0:
            signal[i, :] = (signal[i, :] - signal[i, :].mean()) / (signal[i, :].std() + 1e-8)
    
    # 5. Pad or clip to seq_length (deterministic center-pad for preprocessing)
    if signal.shape[1] >= seq_length:
        signal = signal[:, :seq_length]
    else:
        pad_total = seq_length - signal.shape[1]
        left = pad_total // 2
        right = pad_total - left
        signal = np.hstack([
            np.zeros((signal.shape[0], left), dtype=np.float32),
            signal,
            np.zeros((signal.shape[0], right), dtype=np.float32)
        ])
    
    return signal.astype(np.float32)


def load_header(header_file):
    """Parse a .hea file to get sampling frequency and SNOMED CT diagnosis codes."""
    with open(header_file, 'r') as f:
        lines = f.readlines()
    
    header_info = lines[0].strip().split()
    fs = int(header_info[2])
    
    diagnoses = []
    for line in lines:
        if '#Dx:' in line.replace(' ', ''):
            dx_str = line.split('Dx:')[1].strip()
            codes = dx_str.split(',')
            for code in codes:
                code = code.strip()
                if code:
                    diagnoses.append(code)
            break
    
    return fs, diagnoses


# Standard CINC 2020 Equivalent Classes (27 scored classes merged into 24)
EQUIVALENT_CLASSES = {
    "713427006": "59118001",   # CRBBB -> RBBB
    "284470004": "63593006",   # PAC -> SVPB
    "427172004": "17338001"    # PVC -> VPB
}


def main():
    parser = argparse.ArgumentParser(description="Pre-process CINC2020 dataset into a single .pt file")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/gundetisiddarth/physionet-2020/phsionet_2020/', help="Root path of CINC2020 dataset")
    parser.add_argument('--target_fs', type=int, default=257, help="Target sampling frequency")
    parser.add_argument('--seq_length', type=int, default=4096, help="Target sequence length")
    parser.add_argument('--out_dir', type=str, default='/kaggle/working/preprocessed', help="Output directory for preprocessed data")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    target_fs = args.target_fs
    seq_length = args.seq_length
    out_dir = args.out_dir
    
    out_path = os.path.join(out_dir, f'cinc2020_12lead_{target_fs}hz_{seq_length}len.pt')
    
    if os.path.exists(out_path):
        print(f"Preprocessed file already exists at: {out_path}")
        print("Skipping preprocessing. Delete this file to regenerate.")
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load scored classes mapping for label encoding
    import pandas as pd
    mapping_path = os.path.join(data_dir, 'dx_mapping_scored.csv')
    dx_df = pd.read_csv(mapping_path)
    scored_classes = dx_df['SNOMED CT Code'].astype(str).tolist()
    
    mapped_classes = set()
    for code in scored_classes:
        mapped_classes.add(EQUIVALENT_CLASSES.get(code, code))
    target_classes = sorted(list(mapped_classes))
    num_classes = len(target_classes)
    class_to_idx = {str(code): i for i, code in enumerate(target_classes)}
    
    print(f"Number of scored classes (after merging): {num_classes}")
    print(f"Target fs: {target_fs} Hz | Sequence length: {seq_length}")
    
    # Collect all record paths from train/test splits across all folds
    folds_dir = os.path.join(data_dir, '5folds')
    all_filenames = set()
    for split_idx in range(5):
        for prefix in ['train_split', 'test_split']:
            csv_path = os.path.join(folds_dir, f'{prefix}{split_idx}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                all_filenames.update(df['filename'].tolist())
    
    all_filenames = sorted(list(all_filenames))
    total = len(all_filenames)
    print(f"Total unique records to preprocess: {total}")
    
    # Pre-allocate tensors
    all_signals = torch.zeros(total, 12, seq_length, dtype=torch.float32)
    all_labels = torch.zeros(total, num_classes, dtype=torch.float32)
    all_fnames = []
    skipped = 0
    
    for idx, fname in enumerate(tqdm(all_filenames, desc="Preprocessing CINC2020")):
        raw_path = fname.replace('\\', '/')
        if 'training/' in raw_path:
            relative_path = raw_path[raw_path.find('training/'):]
        else:
            relative_path = raw_path
        
        record_path = os.path.join(data_dir, relative_path)
        if record_path.endswith('.mat'):
            record_path = record_path[:-4]
        
        hea_file = record_path + '.hea'
        mat_file = record_path + '.mat'
        
        if not os.path.exists(hea_file) or not os.path.exists(mat_file):
            skipped += 1
            all_fnames.append(fname)
            continue
        
        try:
            # Load and resample
            fs, diagnoses = load_header(hea_file)
            x = loadmat(mat_file)
            signal = np.asarray(x['val'], dtype=np.float64)
            
            if fs != target_fs:
                signal = Resample(signal, fs, target_fs)
            
            # Apply all deterministic transforms + pad/clip
            signal = apply_fixed_transforms(signal, seq_length)
            
            # Store signal
            all_signals[idx] = torch.from_numpy(signal)
            
            # Compute multi-hot label
            for dx in diagnoses:
                mapped_dx = EQUIVALENT_CLASSES.get(dx, dx)
                if mapped_dx in class_to_idx:
                    all_labels[idx, class_to_idx[mapped_dx]] = 1.0
            
            all_fnames.append(fname)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            skipped += 1
            all_fnames.append(fname)
    
    print(f"Successfully preprocessed {total - skipped}/{total} records ({skipped} skipped)")
    
    # Build filename-to-index mapping for fast lookup
    fname_to_idx = {fname: i for i, fname in enumerate(all_fnames)}
    
    # Save everything
    save_dict = {
        'signals': all_signals,
        'labels': all_labels,
        'filenames': all_fnames,
        'fname_to_idx': fname_to_idx,
        'target_fs': target_fs,
        'seq_length': seq_length,
        'num_classes': num_classes,
        'target_classes': target_classes,
        'class_to_idx': class_to_idx
    }
    
    print(f"Saving preprocessed data to: {out_path}")
    print(f"  Signals tensor shape: {all_signals.shape}")
    print(f"  Labels tensor shape: {all_labels.shape}")
    file_size_gb = all_signals.element_size() * all_signals.nelement() / (1024**3)
    print(f"  Estimated file size: ~{file_size_gb:.2f} GB")
    
    torch.save(save_dict, out_path)
    print("Done! Preprocessing complete.")


if __name__ == '__main__':
    main()
