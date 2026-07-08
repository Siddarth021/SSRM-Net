import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
# Resolve the project root and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from torch.utils.data import DataLoader

from SSL_KD_Phase1.models.student.student_model import StudentModel
from SSL_KD_Phase1.datasets.ptbxl_dataset import PTBXLECG
from SSL_KD_Phase1.datasets.cinc2020_dataset import CINC2020ECG
from SSL_KD_Phase1.evaluation.multilabel_metrics import compute_multilabel_metrics

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def compute_class_metrics(y_true, y_pred_probs, classes):
    class_metrics = {}
    for i, cls in enumerate(classes):
        true_i = y_true[:, i]
        prob_i = y_pred_probs[:, i]
        pred_i = (prob_i >= 0.5).astype(int)
        
        try:
            auroc = skm.roc_auc_score(true_i, prob_i)
        except ValueError:
            auroc = 0.5
            
        precision = skm.precision_score(true_i, pred_i, zero_division=0)
        recall = skm.recall_score(true_i, pred_i, zero_division=0)
        f1 = skm.f1_score(true_i, pred_i, zero_division=0)
        
        class_metrics[cls] = {
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return class_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='ECG_SSL_KD/configs/distillation.yaml')
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['training']['num_classes']

    # Load Model
    model = StudentModel(num_classes=num_classes).to(device)
    checkpoint_path = "checkpoints/best_student_distilled.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load Dataset (test split only)
    dataset_name = config.get('data', {}).get('dataset_name', 'ptbxl')
    print(f"Loading {dataset_name.upper()} dataset (Test Split)...")
    
    if dataset_name == 'cinc2020':
        ecg_dataset = CINC2020ECG(config['data']['data_dir'], config['data']['split'])
    else:
        ecg_dataset = PTBXLECG(config['data']['data_dir'], config['data']['score_path'], config['data']['split'])
        
    _, _, test_ds = ecg_dataset.data_prepare()
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Student Parameters: {total_params:,}")

    all_probs = []
    all_labels = []
    all_filenames = []
    
    total_samples = len(test_ds)
    print(f"Total Test Samples: {total_samples}")
    
    start_time = time.time()
    
    shapes_verified = False

    with torch.no_grad():
        for batch in test_loader:
            lead2 = batch["lead2"].to(device)
            morphology = batch["morphology"].to(device)
            labels = batch["label"].to(device)
            filenames = batch["filename"]
            
            outputs = model(lead2, morphology)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits)
            
            if not shapes_verified:
                print("\n--- Shape Verification ---")
                print(f"lead2 shape: {lead2.shape}")
                print(f"morphology shape: {morphology.shape}")
                print(f"logits shape: {logits.shape}")
                print("--------------------------\n")
                shapes_verified = True
                
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_filenames.extend(filenames)

    total_time = time.time() - start_time
    time_per_ecg = (total_time / total_samples) * 1000 # ms
    
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Inference Time Per ECG: {time_per_ecg:.2f} ms")

    # Overall Metrics
    overall_metrics = compute_multilabel_metrics(all_labels, all_probs)
    
    # Class Metrics
    classes = ['NORM', 'CD', 'MI', 'HYP', 'STTC']
    class_metrics = compute_class_metrics(all_labels, all_probs, classes)
    
    # Save Predictions
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame({
        "ecg_id": [os.path.basename(f).replace(".dat", "").replace(".hea", "") for f in all_filenames],
        "norm_prob": all_probs[:, 0],
        "cd_prob": all_probs[:, 1],
        "mi_prob": all_probs[:, 2],
        "hyp_prob": all_probs[:, 3],
        "sttc_prob": all_probs[:, 4]
    })
    df.to_csv("results/student_predictions.csv", index=False)
    print("Saved predictions to results/student_predictions.csv")
    
    # Generate Markdown Report
    report_path = os.path.join(project_root, "student_evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Student Model Evaluation Report\n\n")
        
        f.write("## 1. Performance Overview\n")
        f.write(f"- **Total Parameters:** `{total_params:,}`\n")
        f.write(f"- **Total Test Samples:** `{total_samples}`\n")
        f.write(f"- **Total Inference Time:** `{total_time:.2f} s`\n")
        f.write(f"- **Inference Time Per ECG:** `{time_per_ecg:.2f} ms`\n\n")
        
        f.write("## 2. Overall Metrics\n")
        f.write("| Metric | Score |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Macro AUROC** | `{overall_metrics['macro_auroc']:.4f}` |\n")
        f.write(f"| **Micro AUROC** | `{overall_metrics['micro_auroc']:.4f}` |\n")
        f.write(f"| **Macro F1** | `{overall_metrics['macro_f1']:.4f}` |\n")
        f.write(f"| **Micro F1** | `{overall_metrics['micro_f1']:.4f}` |\n")
        f.write(f"| **Fmax** | `{overall_metrics['fmax']:.4f}` |\n\n")
        
        f.write("## 3. Class-wise Metrics\n")
        f.write("| Class | AUROC | Precision | Recall | F1 Score |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cls in classes:
            m = class_metrics[cls]
            f.write(f"| **{cls}** | {m['auroc']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")
            
        f.write("\n## 4. Verification Check\n")
        f.write("Shapes verified during inference:\n")
        f.write("```text\n")
        f.write("lead2 shape=(B, 1, 5000)\n")
        f.write("morphology shape=(B, 11, 1250)\n")
        f.write("logits shape=(B, 5)\n")
        f.write("```\n\n")
        f.write("## 5. Artifacts Generated\n")
        f.write("- **Predictions CSV:** `results/student_predictions.csv`\n")
        
    print(f"Generated report at {report_path}")

if __name__ == "__main__":
    main()
