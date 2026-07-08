"""
Genetic Algorithm Threshold Optimization for PhysioNet 2020 Challenge Metric.
Adapted from LRH-Net (Differential Evolution threshold search).

Usage:
    python optimize_thresholds.py --config ../configs/cinc2020_finetune.yaml --checkpoint D:\ECG_SSL_KD\checkpoints\finetune_20260708_082745\best_teacher_finetuned.pth
"""
import os
import sys
import argparse
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import differential_evolution
from sklearn.metrics import precision_recall_curve, roc_curve

# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from SSL_KD.datasets.cinc2020_dataset import CINC2020ECG
from SSL_KD.training.finetune_teacher import TeacherFineTuneModel
from SSL_KD.evaluation.multilabel_metrics import load_weights, compute_challenge_metric


class OptimGeneticsObjective:
    def __init__(self, target, outputs, classes, weights_file):
        self.target = target
        self.outputs = outputs
        self.normal_class = '426783006'
        self.weights = load_weights(weights_file, classes)
        self.classes = classes

    def __call__(self, x):
        # x is the threshold vector (24,)
        # Apply element-wise thresholds
        binary_preds = (self.outputs >= x).astype(int)
        
        # Calculate challenge metric (multiply by -1 to maximize)
        score = compute_challenge_metric(
            self.weights, self.target, binary_preds, self.classes, self.normal_class
        )
        return -score if not np.isnan(score) else 0.0


def find_optimal_thresholds(y_true, y_pred, classes, weights_file, maxiter=15):
    """Genetic algorithm using differential evolution to maximize challenge metric."""
    num_classes = y_true.shape[1]
    f1prcT = np.zeros((num_classes,))
    f1rocT = np.zeros((num_classes,))

    # Find starting heuristics via single-class F1 searches
    print("Calculating initial precision-recall/ROC heuristics...", flush=True)
    for j in range(num_classes):
        # PRC heuristic
        prc, rec, thr = precision_recall_curve(y_true[:, j], y_pred[:, j])
        fscore = 2 * prc * rec / (prc + rec + 1e-8)
        idx = np.nanargmax(fscore)
        f1prcT[j] = thr[idx] if idx < len(thr) else 0.5

        # ROC heuristic
        fpr, tpr, thr = roc_curve(y_true[:, j], y_pred[:, j])
        fscore = 2 * (1 - fpr) * tpr / (1 - fpr + tpr)
        idx = np.nanargmax(fscore)
        f1rocT[j] = thr[idx] if idx < len(thr) else 0.5

    # Initialize population of candidates
    pop_size = 300
    population = np.random.rand(pop_size, num_classes)
    for i in range(1, 99):
        population[i, :] = i / 100

    population[100] = f1rocT
    population[101] = f1prcT
    
    bounds = [(0.01, 0.99) for _ in range(num_classes)]
    
    print("Differential Evolution optimization started...", flush=True)
    objective = OptimGeneticsObjective(y_true, y_pred, classes, weights_file)
    
    result = differential_evolution(
        objective, bounds=bounds, disp=True, init=population, 
        workers=1, maxiter=maxiter, popsize=5
    )
    
    print("Differential Evolution optimization complete!")
    return result.x


def main():
    parser = argparse.ArgumentParser(description="Optimize thresholds for finetuned model")
    parser.add_argument('--config', type=str, required=True, help="Finetuning config path")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best_teacher_finetuned.pth checkpoint")
    parser.add_argument('--maxiter', type=int, default=15, help="Max differential evolution iterations")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['training']['num_classes']
    
    # 1. Load Dataset
    print("Loading validation dataset...")
    target_fs = config['data'].get('target_fs', 500)
    seq_length = config['data'].get('seq_length', 5000)
    lead2_sec = config['data'].get('lead2_sec', 10.0)
    morphology_sec = config['data'].get('morphology_sec', 2.5)
    
    ecg_dataset = CINC2020ECG(
        config['data']['data_dir'], config['data']['split'], 
        target_fs=target_fs, seq_length=seq_length,
        lead2_sec=lead2_sec, morphology_sec=morphology_sec
    )
    _, val_ds, _ = ecg_dataset.data_prepare()
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)
    
    # 2. Instantiate Model and Load Checkpoint
    print(f"Loading finetuned model weights from: {args.checkpoint}")
    model = TeacherFineTuneModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Predict on validation set
    print("Predicting on validation set...")
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(lead2, morphology)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    y_pred = np.vstack(all_probs)
    y_true = np.vstack(all_labels)

    # 4. Challenge Metric classes and weight setup
    dx_mapping_path = os.path.join(config['data']['data_dir'], 'dx_mapping_scored.csv')
    weights_file = os.path.join(config['data']['data_dir'], 'weights.csv')
    label_file = pd_labels = np.zeros((1, 24)) # dummy read to fetch target classes
    
    # Extract equivalent classes matching dataset file
    label_df = yaml_label_file = np.zeros((1, 24))
    import pandas as pd
    label_df = pd.read_csv(dx_mapping_path)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_df['SNOMED CT Code']]) - set(equivalent_classes)))
    
    weights = load_weights(weights_file, classes)
    indices = np.any(weights, axis=0)
    classes_scored = [x for i, x in enumerate(classes) if indices[i]]
    
    y_true_scored = y_true[:, indices]
    y_pred_scored = y_pred[:, indices]

    # Calculate score at standard 0.5 threshold
    default_preds = (y_pred_scored >= 0.5).astype(int)
    default_score = compute_challenge_metric(
        weights[np.ix_(indices, indices)], y_true_scored, default_preds, classes_scored, '426783006'
    )
    print(f"\nDefault Threshold (0.5) Challenge Metric Score: {default_score:.4f}")

    # 5. Optimize thresholds
    optimal_thresholds = find_optimal_thresholds(
        y_true_scored, y_pred_scored, classes_scored, weights_file, maxiter=args.maxiter
    )
    
    # Calculate optimized score
    opt_preds = (y_pred_scored >= optimal_thresholds).astype(int)
    opt_score = compute_challenge_metric(
        weights[np.ix_(indices, indices)], y_true_scored, opt_preds, classes_scored, '426783006'
    )
    
    # Map back to full 24 classes (unscored classes get default 0.5)
    full_thresholds = np.ones((num_classes,)) * 0.5
    full_thresholds[indices] = optimal_thresholds

    print(f"\n==========================================")
    print(f"Original CM Score (Threshold=0.5): {default_score:.4f}")
    print(f"Optimized CM Score:               {opt_score:.4f}")
    print(f"Improvement:                      +{opt_score - default_score:.4f}")
    print(f"==========================================\n")

    # 6. Save optimized thresholds
    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "optimized_thresholds.npz")
    np.savez(out_path, thresholds=full_thresholds)
    print(f"Optimized thresholds saved successfully to: {out_path}")


if __name__ == '__main__':
    main()
