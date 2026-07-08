import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from SSL_KD_Phase1.models.teacher.teacher_model import TeacherModel
from SSL_KD_Phase1.ssl.projection_head import ProjectionHead
from SSL_KD_Phase1.ssl.augmentations import ComposeSSL, RandomMask, GaussianNoise, AmplitudeScaling, TimeShift, LeadDropout
from SSL_KD_Phase1.ssl.contrastive_loss import NTXentLoss
from SSL_KD_Phase1.ssl.ssl_trainer import SSLTrainer

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_verify():
    print("Starting SSL training verification step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate models
    teacher = TeacherModel().to(device)
    projection_head = ProjectionHead().to(device)
    
    # Set up augmentations
    augmentations = ComposeSSL([
        RandomMask(0.1, 0.3),
        GaussianNoise(0.02),
        AmplitudeScaling(0.8, 1.2),
        TimeShift(0.05),
        LeadDropout(0.3)
    ])
    
    loss_fn = NTXentLoss(temperature=0.1)
    
    params = list(teacher.parameters()) + list(projection_head.parameters())
    optimizer = optim.Adam(params, lr=0.0003)
    
    trainer = SSLTrainer(teacher, projection_head, augmentations, loss_fn, optimizer, device)
    
    # 1 Train batch
    lead2 = torch.randn(8, 1, 4096).to(device)
    morphology = torch.randn(8, 11, 642).to(device)
    
    loss_val = trainer.train_step(lead2, morphology)
    
    print(f"Validation step loss: {loss_val}")
    
    # Checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(teacher.state_dict(), "checkpoints/best_ssl_teacher.pth")
    torch.save(projection_head.state_dict(), "checkpoints/best_projection_head.pth")
    
    print("Checkpoints saved successfully.")
    print("SSL Trainer verified successfully.")


def main():
    parser = argparse.ArgumentParser(description="SSL Pretraining")
    parser.add_argument('--config', type=str, default='ECG_SSL_KD/configs/ssl.yaml', help="Config file path")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint_latest.pth to resume training")
    parser.add_argument('--verify', action='store_true', help="Run verification mode")
    args = parser.parse_args()

    if args.verify:
        run_verify()
        return

    config = load_yaml(args.config)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    # Initialize Models
    teacher = TeacherModel().to(device)
    projection_head = ProjectionHead().to(device)
    
    augmentations = ComposeSSL([
        RandomMask(0.1, 0.3),
        GaussianNoise(0.02),
        AmplitudeScaling(0.8, 1.2),
        TimeShift(0.05),
        LeadDropout(0.3)
    ])
    
    temp_val = config.get('ssl', {}).get('temperature', config.get('training', {}).get('temperature', 0.1))
    loss_fn = NTXentLoss(temperature=temp_val)
    
    params = list(teacher.parameters()) + list(projection_head.parameters())
    optimizer = optim.Adam(params, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    scaler = GradScaler('cuda')
    trainer = SSLTrainer(teacher, projection_head, augmentations, loss_fn, optimizer, device, scaler=scaler)
    
    import time
    from datetime import datetime
    import shutil

    # Setup unique checkpoint run folder
    save_dir = config.get('checkpoints', {}).get('save_dir', 'checkpoints/')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"ssl_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_path = os.path.join(run_dir, "train.log")

    def log_print(msg):
        print(msg, flush=True)
        with open(log_path, "a") as lf:
            lf.write(msg + "\n")
        # Also append to the master pipeline log so user can monitor externally
        master_log = r"D:\ECG_SSL_KD\SSL_KD\training\pipeline_monitor.log"
        if os.path.exists(master_log) or not args.resume:
            with open(master_log, "a") as ml:
                ml.write(msg + "\n")

    # Copy the config file used into the run folder (if not resuming)
    if not args.resume:
        shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    dataset_name = config.get('data', {}).get('dataset_name', 'ptbxl')
    log_print(f"Loading {dataset_name.upper()} dataset...")
    
    if dataset_name == 'cinc2020':
        from SSL_KD_Phase1.datasets.cinc2020_dataset import CINC2020ECG
        target_fs = config['data'].get('target_fs', 257)
        seq_length = config['data'].get('seq_length', 4096)
        lead2_sec = config['data'].get('lead2_sec', 15.94)
        morphology_sec = config['data'].get('morphology_sec', 2.5)
        
        ecg_dataset = CINC2020ECG(
            config['data']['data_dir'], config['data']['split'],
            target_fs=target_fs, seq_length=seq_length,
            lead2_sec=lead2_sec, morphology_sec=morphology_sec
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is no longer supported.")
        
    train_ds, val_ds, test_ds = ecg_dataset.data_prepare()
    
    # Retrieve a sample for shape verification
    sample = train_ds[0]
    lead2_shape = sample["lead2"].shape
    morphology_shape = sample["morphology"].shape
    label_shape = sample["label"].shape if "label" in sample else "N/A"
    
    log_print(f"Dataset root path: {config['data']['data_dir']}")
    log_print(f"Number of ECG records: {len(train_ds) + len(val_ds) + len(test_ds)}")
    log_print(f"Train samples: {len(train_ds)}")
    log_print(f"Validation samples: {len(val_ds)}")
    log_print(f"Test samples: {len(test_ds)}")
    log_print(f"Lead2 shape: {lead2_shape}")
    log_print(f"Morphology shape: {morphology_shape}")
    log_print(f"Label shape: {label_shape}")
    
    num_workers = config['training'].get('num_workers', 4)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    best_loss = float('inf')
    epochs = config['training']['epochs']
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        log_print(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        teacher.load_state_dict(checkpoint['teacher_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        # Override run_dir to save in the same directory as the resume checkpoint
        run_dir = os.path.dirname(args.resume)
        log_path = os.path.join(run_dir, "train.log")
        log_print(f"Resumed at epoch {start_epoch} with best_loss {best_loss:.6f}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Wrap train_loader with tqdm for the progress bar
        train_pbar = tqdm(train_loader, desc=f"Phase 0 | Epoch {epoch}/{epochs}")
        for batch in train_pbar:
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            loss_val = trainer.train_step(lead2, morphology)
            epoch_loss += loss_val * lead2.size(0)
            
            # Update the progress bar text dynamically
            train_pbar.set_postfix({'loss': f"{loss_val:.4f}"})
            
        epoch_loss /= len(train_loader.dataset)
        
        epoch_duration = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        log_print(f"epoch={epoch} | ssl_loss={epoch_loss:.6f} | learning_rate={current_lr:.6f} | time_taken={epoch_duration:.2f}s")
        
        # Save checkpoints
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(teacher.state_dict(), os.path.join(run_dir, "best_ssl_teacher.pth"))
            torch.save(projection_head.state_dict(), os.path.join(run_dir, "best_projection_head.pth"))
            
        # Save latest state for resuming
        latest_state = {
            'epoch': epoch,
            'teacher_state_dict': teacher.state_dict(),
            'projection_head_state_dict': projection_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }
        torch.save(latest_state, os.path.join(run_dir, "checkpoint_latest.pth"))

if __name__ == '__main__':
    main()
