import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import sklearn, scipy, pandas first to avoid Windows MKL/OpenMP conflicts with PyTorch
import sklearn.metrics as skm
import pandas as pd
import scipy

import sys
# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import argparse
import yaml
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from SSL_KD_Phase1.training.finetune_teacher import TeacherFineTuneModel
from SSL_KD_Phase1.models.student.student_model import StudentModel
from SSL_KD_Phase1.distillation.distillation_loss import DistillationLoss

from SSL_KD_Phase1.datasets.cinc2020_dataset import CINC2020ECG
from SSL_KD_Phase1.evaluation.multilabel_metrics import compute_multilabel_metrics

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Student Knowledge Distillation")
    parser.add_argument('--config', type=str, default='ECG_SSL_KD/configs/distillation.yaml', help="Config file path")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint_latest.pth to resume training")
    parser.add_argument('--verify', action='store_true', help="Run verification mode")
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    num_classes = config['training']['num_classes']

    # Setup unique checkpoint run folder
    save_dir = config['checkpoints']['save_dir']
    run_dir = os.path.join(save_dir, f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_path = os.path.join(run_dir, "distill.log")

    def log_print(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
        # Also append to the master pipeline log so user can monitor externally
        master_log = r"D:\ECG_SSL_KD\SSL_KD\training\pipeline_monitor.log"
        if os.path.exists(master_log) or not args.resume:
            with open(master_log, "a", encoding="utf-8") as ml:
                ml.write(msg + "\n")

    # Copy the config file used into the run folder (if not resuming)
    if not args.resume:
        shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    log_print("=== Knowledge Distillation Stage ===")
    log_print(f"Device: {device}")

    # 1. Instantiate Teacher and Student Models
    teacher_model = TeacherFineTuneModel(num_classes=num_classes).to(device)
    teacher_checkpoint = config['checkpoints']['teacher_checkpoint']
    
    if os.path.exists(teacher_checkpoint):
        teacher_model.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
        log_print(f"Teacher model loaded successfully from {teacher_checkpoint}")
    else:
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint}. Train teacher first.")
        
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = StudentModel(num_classes=num_classes).to(device)

    # 2. Print parameters and compression ratio
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params

    log_print(f"Teacher parameters: {teacher_params:,}")
    log_print(f"Student parameters: {student_params:,}")
    log_print(f"Compression ratio: {compression_ratio:.2f}x")

    # 3. Instantiate Loss and Optimizer
    distillation_criterion = DistillationLoss(
        temperature=config['training']['temperature'],
        alpha=config['training']['alpha'],
        beta=config['training']['beta'],
        gamma=config['training']['gamma']
    )
    
    optimizer = optim.Adam(
        student_model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )

    # 4. Load Dataset
    dataset_name = config.get('data', {}).get('dataset_name', 'ptbxl')
    log_print(f"Loading {dataset_name.upper()} dataset...")
    if dataset_name == 'cinc2020':
        target_fs = config['data'].get('target_fs', 500)
        seq_length = config['data'].get('seq_length', 5000)
        lead2_sec = config['data'].get('lead2_sec', 10.0)
        morphology_sec = config['data'].get('morphology_sec', 2.5)
        ecg_dataset = CINC2020ECG(
            config['data']['data_dir'], config['data']['split'], 
            target_fs=target_fs, seq_length=seq_length,
            lead2_sec=lead2_sec, morphology_sec=morphology_sec
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is no longer supported.")
    train_ds, val_ds, test_ds = ecg_dataset.data_prepare()
    
    log_print(f"Train samples: {len(train_ds)}")
    log_print(f"Validation samples: {len(val_ds)}")
    log_print(f"Test samples: {len(test_ds)}")

    num_workers = config['training'].get('num_workers', 4)
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    # 5. Training loop
    epochs = config['training']['epochs']
    
    monitor_metric = config.get('training', {}).get('monitor_metric', 'macro_auroc')
    log_print(f"Monitoring metric for checkpointing: {monitor_metric}")
    best_score = float('inf') if monitor_metric == 'val_loss' else -1.0
    
    best_epoch = 0
    best_metrics = {}
    start_epoch = 1

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')

    if args.resume and os.path.exists(args.resume):
        log_print(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', float('inf') if monitor_metric == 'val_loss' else -1.0)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            scheduler.last_epoch = start_epoch - 1
        # Override run_dir
        run_dir = os.path.dirname(args.resume)
        log_path = os.path.join(run_dir, "distill.log")
        log_print(f"Resumed at epoch {start_epoch} with best_score {best_score:.6f}")

    training_start_time = time.time()

    log_print("Starting Student Distillation Training Loop...")
    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        student_model.train()
        
        train_loss = 0.0
        train_bce = 0.0
        train_mse = 0.0
        train_kl = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass teacher (frozen)
            with torch.no_grad():
                teacher_embedding = teacher_model.encoder(lead2, morphology)["embedding"]
                teacher_logits = teacher_model.classifier(teacher_embedding)
            
            # Forward pass student with mixed precision
            with autocast('cuda'):
                student_outputs = student_model(lead2, morphology)
                
                if batch_idx == 0:
                    log_print(f"  [Epoch {epoch} First Batch] teacher_embedding shape={teacher_embedding.shape}")
                    log_print(f"  [Epoch {epoch} First Batch] student_projection shape={student_outputs['projected'].shape}")
                    raw_mse = torch.nn.functional.mse_loss(student_outputs["projected"], teacher_embedding)
                    log_print(f"  [Epoch {epoch} First Batch] raw_mse.item() = {raw_mse.item():.8f}")
                
                # Calculate distillation loss
                loss_dict = distillation_criterion(
                    student_outputs["logits"],
                    student_outputs["projected"],
                    teacher_logits,
                    teacher_embedding,
                    labels
                )
                
                loss = loss_dict["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * lead2.size(0)
            train_bce += loss_dict["bce"].item() * lead2.size(0)
            train_mse += loss_dict["feature_mse"].item() * lead2.size(0)
            train_kl += loss_dict["kl_div"].item() * lead2.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_bce /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        train_kl /= len(train_loader.dataset)

        # Validation phase
        student_model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                lead2 = batch["lead2"].to(device)
                morphology = batch["morphology"].to(device)
                labels = batch["label"].to(device)
                
                # We calculate standard distillation loss during validation to monitor components
                teacher_embedding = teacher_model.encoder(lead2, morphology)["embedding"]
                teacher_logits = teacher_model.classifier(teacher_embedding)
                
                student_outputs = student_model(lead2, morphology)
                
                loss_dict = distillation_criterion(
                    student_outputs["logits"],
                    student_outputs["projected"],
                    teacher_logits,
                    teacher_embedding,
                    labels
                )
                
                val_loss += loss_dict["loss"].item() * lead2.size(0)
                
                probs = torch.sigmoid(student_outputs["logits"])
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                
            val_loss /= len(val_loader.dataset)
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_multilabel_metrics(all_labels, all_probs, data_dir=config['data']['data_dir'])
        epoch_duration = time.time() - epoch_start_time
        
        log_print(f"Epoch: {epoch}/{epochs} | "
                  f"Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, MSE: {train_mse:.4f}, KL: {train_kl:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"CM Score: {metrics.get('cm_score', 0.0):.4f} | "
                  f"Macro AUROC: {metrics['macro_auroc']:.4f} | "
                  f"Micro AUROC: {metrics['micro_auroc']:.4f} | "
                  f"Macro F1: {metrics['macro_f1']:.4f} | "
                  f"Micro F1: {metrics['micro_f1']:.4f} | "
                  f"Fmax: {metrics['fmax']:.4f} | "
                  f"Time: {epoch_duration:.1f}s")
        
        current_score = val_loss if monitor_metric == 'val_loss' else metrics.get(monitor_metric, 0.0)
        is_best = False
        if monitor_metric == 'val_loss':
            if current_score < best_score:
                is_best = True
        else:
            if current_score >= best_score:
                is_best = True

        # Save student checkpoint if performance improves
        if is_best:
            best_score = current_score
            best_epoch = epoch
            best_metrics = metrics
            
            # Save whole model state
            torch.save(student_model.state_dict(), os.path.join(run_dir, "best_student_distilled.pth"))
            # Also save to base checkpoints folder
            torch.save(student_model.state_dict(), os.path.join(save_dir, "best_student_distilled.pth"))
            log_print(f"  --> Saved new best student model checkpoint ({monitor_metric}: {best_score:.4f})")

        # Save latest state for resuming
        latest_state = {
            'epoch': epoch,
            'student_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score
        }
        torch.save(latest_state, os.path.join(run_dir, "checkpoint_latest.pth"))

        # Step the learning rate scheduler
        scheduler.step()

    log_print("\n=== Distillation Completed ===")
    log_print(f"Best Student Epoch: {best_epoch}")
    log_print(f"Best Student {monitor_metric.upper()}: {best_score:.4f}")
    log_print(f"Best Student CM Score: {best_metrics.get('cm_score', 0.0):.4f}")
    log_print(f"Best Student Macro AUROC: {best_metrics.get('macro_auroc', 0.0):.4f}")
    log_print(f"Best Student Micro AUROC: {best_metrics.get('micro_auroc', 0.0):.4f}")
    log_print(f"Best Student Macro F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    log_print(f"Best Student Micro F1: {best_metrics.get('micro_f1', 0.0):.4f}")
    log_print(f"Best Student Fmax: {best_metrics.get('fmax', 0.0):.4f}")

if __name__ == '__main__':
    main()
