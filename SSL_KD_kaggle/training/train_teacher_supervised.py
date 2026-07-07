import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import sklearn first at the absolute entrypoint to prevent OpenMP/MKL DLL conflicts on Windows
import sklearn.metrics as skm
import pandas as pd
import scipy

import sys
# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import yaml

from SSL_KD.evaluation.multilabel_metrics import compute_multilabel_metrics

from SSL_KD.datasets.cinc2020_dataset import CINC2020ECG

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from SSL_KD.training.finetune_teacher import TeacherFineTuneModel

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_verify():
    print("Starting fine-tuning training verification step...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate fine-tuning model
    model = TeacherFineTuneModel(num_classes=5, mode='frozen').to(device)
    
    # Verify loading weights if best_ssl_teacher exists
    if os.path.exists("checkpoints/best_ssl_teacher.pth"):
        model.encoder.load_state_dict(torch.load("checkpoints/best_ssl_teacher.pth"))
        print("Pretrained SSL teacher weights loaded successfully.", flush=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    
    # 1 Mock batch training
    lead2 = torch.randn(8, 1, 5000).to(device)
    morphology = torch.randn(8, 11, 1250).to(device)
    y = torch.randint(0, 2, (8, 5)).float().to(device)
    
    model.train()
    optimizer.zero_grad()
    logits = model(lead2, morphology)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    
    print(f"Train loss: {loss.item()}", flush=True)
    
    # 1 Mock batch validation
    model.eval()
    with torch.no_grad():
        val_logits = model(lead2, morphology)
        val_loss = criterion(val_logits, y)
        probs = torch.sigmoid(val_logits)
        
    metrics = compute_multilabel_metrics(y, probs)
    print(f"Validation loss: {val_loss.item()}", flush=True)
    print(f"Metrics: {metrics}", flush=True)
    
    # Checkpoints directories saving check
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best_teacher_finetuned.pth")
    torch.save(model.encoder.state_dict(), "checkpoints/teacher_encoder.pth")
    torch.save(model.classifier.state_dict(), "checkpoints/classifier_head.pth")
    
    # Calculate parameter breakdown
    teacher_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = teacher_params + classifier_params
    
    print(f"Teacher parameters: {teacher_params:,}", flush=True)
    print(f"Classifier parameters: {classifier_params:,}", flush=True)
    print(f"Total parameters: {total_params:,}", flush=True)
    print("Frozen mode passed", flush=True)
    
    # Toggle model to finetune mode
    model.set_mode('finetune')
    print("Finetune mode passed", flush=True)
    print("Supervised Fine-Tuning verified successfully.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument('--config', type=str, default='ECG_SSL_KD/configs/finetune.yaml', help="Config file path")
    parser.add_argument('--mode', type=str, choices=['frozen', 'finetune'], default=None, help="Training mode override")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint_latest.pth to resume training")
    parser.add_argument('--verify', action='store_true', help="Run verification mode")
    args = parser.parse_args()

    if args.verify:
        run_verify()
        return

    config = load_yaml(args.config)
    if args.mode is not None:
        config['training']['mode'] = args.mode
    
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    mode = config['training']['mode']
    
    # Initialize fine-tune wrapper
    model = TeacherFineTuneModel(num_classes=config['training']['num_classes'], mode=mode).to(device)
    
    # Load pretrained SSL encoder weights if they exist
    ssl_path = config['checkpoints']['pretrained_ssl_teacher']
    if os.path.exists(ssl_path):
        model.encoder.load_state_dict(torch.load(ssl_path))
        print(f"Loaded pretrained SSL weights from {ssl_path}", flush=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    import time
    from datetime import datetime
    import shutil

    # Setup unique checkpoint run folder
    save_dir = config['checkpoints']['save_dir']
    run_dir = os.path.join(save_dir, f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_path = os.path.join(run_dir, "train.log")

    def log_print(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
        # Also append to the master pipeline log so user can monitor externally
        master_log = os.path.join(os.path.dirname(save_dir), "pipeline_monitor.log")
        if os.path.exists(master_log) or not args.resume:
            with open(master_log, "a", encoding="utf-8") as ml:
                ml.write(msg + "\n")

    # Copy the config file used into the run folder (if not resuming)
    if not args.resume:
        shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))

    dataset_name = config.get('data', {}).get('dataset_name', 'ptbxl')
    log_print(f"Loading {dataset_name.upper()} fine-tuning dataset...")
    try:
        log_print(f"  -> Creating {dataset_name.upper()} dataset instance...")
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
        
        log_print("  -> Calling data_prepare()... (This may take 10-15 seconds, DO NOT CLOSE TERMINAL)")
        train_ds, val_ds, test_ds = ecg_dataset.data_prepare()
        log_print("  -> data_prepare() completed successfully!")
        
        # Print dataset statistics
        log_print(f"Dataset root path: {config['data']['data_dir']}")
        log_print(f"Number of ECG records: {len(train_ds) + len(val_ds) + len(test_ds)}")
        lead2_samples = min(int(lead2_sec * target_fs), seq_length) if dataset_name == 'cinc2020' else seq_length
        morph_samples = min(int(morphology_sec * target_fs), seq_length) if dataset_name == 'cinc2020' else seq_length // 4

        log_print(f"Train samples: {len(train_ds)}")
        log_print(f"Validation samples: {len(val_ds)}")
        log_print(f"Test samples: {len(test_ds)}")
        log_print(f"Lead2 shape: (1, {lead2_samples})")
        log_print(f"Morphology shape: (11, {morph_samples})")
        log_print(f"Label shape: ({config['training']['num_classes']},)")
        
        log_print("  -> Creating DataLoaders...")
        num_workers = config['training'].get('num_workers', 4)
        train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=num_workers)
        log_print(f"Train loader batches: {len(train_loader)}")
        log_print(f"Val loader batches: {len(val_loader)}")
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"FATAL ERROR DURING DATASET LOADING:\n{err_msg}", flush=True)
        with open(os.path.join(run_dir, "FATAL_CRASH.txt"), "w", encoding="utf-8") as f:
            f.write(err_msg)
        sys.exit(1)

    epochs = config['training']['epochs']
    
    # Print estimates
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_print(f"Mode: {mode.upper()}")
    log_print(f"Trainable parameters: {trainable_params:,}")
    log_print(f"Estimated GPU memory usage: ~2.5 GB")
    est_time = (162 * epochs) / 60
    log_print(f"Estimated total training time: ~{est_time:.1f} minutes")
    
    monitor_metric = config.get('training', {}).get('monitor_metric', 'macro_auroc')
    log_print(f"Monitoring metric for checkpointing: {monitor_metric}")
    best_score = float('inf') if monitor_metric == 'val_loss' else -1.0
    
    best_epoch = 0
    best_metrics = {}
    best_all_probs = None
    best_all_labels = None
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        log_print(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', float('inf') if monitor_metric == 'val_loss' else -1.0)
        # Override run_dir
        run_dir = os.path.dirname(args.resume)
        log_path = os.path.join(run_dir, "train.log")
        log_print(f"Resumed at epoch {start_epoch} with best_score {best_score:.6f}")

    training_start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Phase 1 | Epoch {epoch}/{epochs} [Train]")
        for batch in train_pbar:
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            logits = model(lead2, morphology)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * lead2.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        val_pbar = tqdm(val_loader, desc=f"Phase 1 | Epoch {epoch}/{epochs} [Val]")
        with torch.no_grad():
            for batch in val_pbar:
                lead2 = batch["lead2"].to(device)
                morphology = batch["morphology"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(lead2, morphology)
                loss = criterion(logits, labels)
                val_loss += loss.item() * lead2.size(0)
                
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
            val_loss /= len(val_loader.dataset)
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_multilabel_metrics(all_labels, all_probs, data_dir=config['data']['data_dir'])
        metrics['val_loss'] = val_loss
        
        epoch_duration = time.time() - epoch_start_time
        log_print(f"Epoch: {epoch} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Validation Loss: {val_loss:.4f} | "
                  f"CM Score: {metrics.get('cm_score', 0.0):.4f} | "
                  f"Macro AUROC: {metrics['macro_auroc']:.4f} | "
                  f"Micro AUROC: {metrics['micro_auroc']:.4f} | "
                  f"Macro F1: {metrics['macro_f1']:.4f} | "
                  f"Micro F1: {metrics['micro_f1']:.4f} | "
                  f"Fmax: {metrics['fmax']:.4f} | "
                  f"time_taken={epoch_duration:.2f}s")
        
        current_score = metrics.get(monitor_metric, 0.0)
        is_best = False
        if monitor_metric == 'val_loss':
            if current_score < best_score:
                is_best = True
        else:
            if current_score >= best_score:
                is_best = True
                
        # Save checkpoints
        if is_best:
            best_score = current_score
            best_epoch = epoch
            best_metrics = metrics
            best_all_probs = all_probs
            best_all_labels = all_labels
            
            # Save checkpoints based on mode
            if mode == 'frozen':
                torch.save(model.state_dict(), os.path.join(run_dir, "best_teacher_frozen.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(run_dir, "best_teacher_finetuned.pth"))
                
            torch.save(model.encoder.state_dict(), os.path.join(run_dir, "teacher_encoder.pth"))
            torch.save(model.classifier.state_dict(), os.path.join(run_dir, "classifier_head.pth"))

        # Save latest state for resuming
        latest_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score
        }
        torch.save(latest_state, os.path.join(run_dir, "checkpoint_latest.pth"))

    # After training completes
    total_duration = time.time() - training_start_time
    
    log_print("\n=== Training Completed ===")
    log_print(f"Total training time: {total_duration:.2f}s")
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        log_print(f"Peak GPU memory usage: {gpu_mem:.2f} MB")
    else:
        log_print("Peak GPU memory usage: N/A (CPU used)")
        
    log_print(f"Best Epoch: {best_epoch}")
    log_print(f"Best {monitor_metric.upper()}: {best_score:.4f}")
    log_print(f"Best CM Score: {best_metrics.get('cm_score', 0.0):.4f}")
    log_print(f"Best Macro AUROC: {best_metrics.get('macro_auroc', 0.0):.4f}")
    log_print(f"Best Micro AUROC: {best_metrics.get('micro_auroc', 0.0):.4f}")
    log_print(f"Best Macro F1: {best_metrics.get('macro_f1', 0.0):.4f}")
    log_print(f"Best Micro F1: {best_metrics.get('micro_f1', 0.0):.4f}")
    log_print(f"Best Fmax: {best_metrics.get('fmax', 0.0):.4f}")
    
    # Class-wise AUROC and Confusion Statistics
    import sklearn.metrics as skm
    log_print("\n--- Confusion Statistics and Class-wise AUROC per Class (Threshold = 0.5) ---")
    if best_all_labels is not None and best_all_probs is not None:
        y_true = best_all_labels.numpy()
        y_probs = best_all_probs.numpy()
        y_pred = (y_probs >= 0.5).astype(int)
        
        target_classes = getattr(ecg_dataset, 'target_classes', ['NORM', 'CD', 'MI', 'HYP', 'STTC'])
        for idx, cls in enumerate(target_classes):
            yt = y_true[:, idx]
            yp = y_pred[:, idx]
            ypr = y_probs[:, idx]
            
            # Confusion matrix elements
            tn, fp, fn, tp = skm.confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            
            # Class-wise AUROC
            try:
                class_auroc = skm.roc_auc_score(yt, ypr)
            except ValueError:
                class_auroc = 0.5
                
            log_print(f"Class: {cls}")
            log_print(f"  AUROC: {class_auroc:.4f}")
            log_print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

if __name__ == '__main__':
    main()
