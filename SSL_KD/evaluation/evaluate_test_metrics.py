import os
import sys
import json
import time
import argparse
import torch
import numpy as np
import sklearn.metrics as skm
from torch.utils.data import DataLoader

# ========================================================
# ADDED INSTANT FEEDBACK SO SCRIPT DOES NOT APPEAR FROZEN
print("Booting up evaluation script... Please wait ~15s for PyTorch and Metadata to load.")
# ========================================================

# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from SSL_KD.models.student.student_model import StudentModel
from SSL_KD.models.teacher.teacher_model import TeacherModel
from SSL_KD.datasets.ptbxl_dataset import PTBXLECG
from SSL_KD.datasets.cinc2020_dataset import CINC2020ECG
from SSL_KD.evaluation.multilabel_metrics import compute_multilabel_metrics

def compute_class_metrics(y_true, y_pred_probs, classes):
    class_metrics = {}
    for i, cls in enumerate(classes):
        true_i = y_true[:, i]
        prob_i = y_pred_probs[:, i]
        pred_i = (prob_i >= 0.5).astype(int)
        
        try:
            auroc = float(skm.roc_auc_score(true_i, prob_i))
        except ValueError:
            auroc = 0.5
            
        precision = float(skm.precision_score(true_i, pred_i, zero_division=0))
        recall = float(skm.recall_score(true_i, pred_i, zero_division=0))
        f1 = float(skm.f1_score(true_i, pred_i, zero_division=0))
        
        class_metrics[cls] = {
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return class_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model .pth file")
    parser.add_argument('--model_type', type=str, choices=['student', 'teacher'], default='student', help="Type of model")
    parser.add_argument('--data_dir', type=str, default='D:\\Datasets\\PTBXL\\', help="PTB-XL dataset directory")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for inference")
    parser.add_argument('--dataset', type=str, choices=['ptbxl', 'cinc2020'], default='ptbxl', help="Dataset to evaluate on")
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Device to run inference on")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at: {args.model_path}")

    # FORCE CPU IF REQUESTED (or if CUDA is not available)
    if args.device == 'cpu' or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU for inference. No data will be loaded into VRAM.")
    else:
        device = torch.device("cuda")
        print("Using CUDA for inference.")

    if args.dataset == 'cinc2020':
        num_classes = 24
    else:
        num_classes = 5

    # Load Model
    if args.model_type == 'student':
        model = StudentModel(num_classes=num_classes)
    else:
        model = TeacherModel(num_classes=num_classes)
        
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    # Load Dataset (test split only)
    print(f"Loading {args.dataset.upper()} dataset (Test Split) from {args.data_dir}...")
    
    if args.dataset == 'cinc2020':
        ecg_dataset = CINC2020ECG(args.data_dir, 'test')
    else:
        ecg_dataset = PTBXLECG(args.data_dir, os.path.join(args.data_dir, "ptbxl_database.csv"), 'test')
        
    _, _, test_ds = ecg_dataset.data_prepare()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_probs = []
    all_labels = []
    
    total_samples = len(test_ds)
    print(f"Total Test Samples: {total_samples}")
    
    start_time = time.time()
    total_batches = len(test_loader)

    print("\n--- Starting Inference ---")
    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            labels = batch["label"] # Keep labels on CPU
            
            if args.model_type == 'student':
                outputs = model(lead2, morphology)
                logits = outputs["logits"]
            else:
                logits = model(lead2, morphology)
                
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Simple Progress Bar
            if i % max(1, total_batches // 10) == 0 or i == total_batches:
                progress_pct = (i / total_batches) * 100
                print(f"Processed batch {i}/{total_batches} [{progress_pct:.0f}%]")

    total_time = time.time() - start_time
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # Metrics
    overall_metrics = compute_multilabel_metrics(all_labels, all_probs)
    
    if args.dataset == 'cinc2020':
        classes = ecg_dataset.target_classes if hasattr(ecg_dataset, 'target_classes') else [f'C{i}' for i in range(24)]
    else:
        classes = ['NORM', 'CD', 'MI', 'HYP', 'STTC']
        
    class_metrics = compute_class_metrics(all_labels, all_probs, classes)
    
    # Save JSON in the same directory as the model
    save_dir = os.path.dirname(args.model_path)
    if not save_dir:
        save_dir = "."
        
    out_json = os.path.join(save_dir, "test_metrics.json")
    
    output_data = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "inference_time_seconds": round(total_time, 2),
        "overall_metrics": overall_metrics,
        "class_metrics": class_metrics
    }
    
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nEvaluation Complete!")
    print(f"Macro AUROC: {overall_metrics['macro_auroc']:.4f}")
    print(f"Saved metrics to: {out_json}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n!!! SCRIPT CRASHED !!!")
        print(repr(e))
        traceback.print_exc(file=sys.stdout)
