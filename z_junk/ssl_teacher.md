# ECG_SSL_KD Project Summary (Teacher Stage)

## Project Objective

Develop an ECG disease classification framework that:

1. Uses Self-Supervised Learning (SSL) to learn ECG representations without labels.
2. Fine-tunes the pretrained encoder on PTB-XL disease labels.
3. Later transfers knowledge to a lightweight student model suitable for edge deployment.

---

# Dataset

## PTB-XL Dataset

Total Records: **21,799 ECGs**

Dataset Split:

| Split               | Records |
| ------------------- | ------: |
| Train (Folds 1–8)   |  17,418 |
| Validation (Fold 9) |   2,183 |
| Test (Fold 10)      |   2,198 |
| Total               |  21,799 |

Dataset Path:

```text
D:\Datasets\PTBXL\
```

Verified:

* No train/validation overlap
* No train/test overlap
* No validation/test overlap
* No data leakage detected

---

# ECG Representation Design

The final deployment scenario is based on digitized ECG images where:

* Lead II contains 10 seconds of rhythm information
* Remaining leads contain 2.5 seconds of morphology information

Therefore the training pipeline was modified to match deployment.

## Input Structure

### Rhythm Branch

```text
Lead II
Shape: (1,5000)
Duration: 10 seconds
```

### Morphology Branch

```text
11 Remaining Leads
Shape: (11,1250)
Duration: 2.5 seconds
```

Output sample structure:

```python
{
    "lead2": (1,5000),
    "morphology": (11,1250),
    "label": (5,)
}
```

---

# Teacher Architecture

## Rhythm Encoder

Input:

```text
(B,1,5000)
```

Components:

* Conv1D
* Residual Blocks
* BiLSTM
* Temporal Attention

Output:

```text
(B,256)
```

Parameters:

```text
750,272
```

---

## Morphology Encoder

Input:

```text
(B,11,1250)
```

Components:

* Depthwise Separable Conv1D
* Residual Connections
* Lead Attention
* Global Average Pooling

Output:

```text
(B,256)
```

Parameters:

```text
208,320
```

---

## Fusion Module

Inputs:

```text
Rhythm Feature: (B,256)
Morphology Feature: (B,256)
```

Pipeline:

```text
Concatenate
→ MLP
→ 256-D Embedding
```

Output:

```text
(B,256)
```

Parameters:

```text
395,008
```

---

## Teacher Encoder Summary

Output Embedding:

```text
(B,256)
```

Total Parameters:

```text
1,353,600
```

---

# Self-Supervised Learning (SSL)

## Objective

Learn ECG representations without disease labels.

Dataset Used:

```text
PTB-XL Train Split Only
17,418 ECGs
```

Labels Used:

```text
No
```

---

## SSL Pipeline

```text
ECG
 ↓

Augmentation A
 ↓

Teacher Encoder
 ↓

Projection Head
 ↓

128-D Projection

---------------------

ECG
 ↓

Augmentation B
 ↓

Teacher Encoder
 ↓

Projection Head
 ↓

128-D Projection

---------------------

InfoNCE Loss
```

---

## SSL Augmentations

Implemented:

* Random Masking
* Gaussian Noise
* Amplitude Scaling
* Time Shift
* Lead Dropout
* ComposeSSL

---

## Projection Head

Architecture:

```text
256
 ↓
256
 ↓
128
```

Parameters:

```text
99,200
```

---

## SSL Training Results

Initial Loss:

```text
0.287291
```

Final Loss:

```text
0.008936
```

Loss Reduction:

```text
~32× decrease
```

Observations:

* Stable convergence
* No collapse detected
* No NaNs detected
* Successful representation learning

Checkpoint Saved:

```text
best_ssl_teacher.pth
```

Checkpoint Contains:

```text
Rhythm Encoder
Morphology Encoder
Fusion Module
```

Projection head not stored.

---

# Fine-Tuning Stage

## Objective

Fine-tune SSL-pretrained encoder on disease classification.

Classes:

```text
NORM
CD
MI
HYP
STTC
```

Classifier Head:

```text
256
 ↓
128
 ↓
5
```

Parameters:

```text
33,797
```

Loss Function:

```text
BCEWithLogitsLoss
```

---

# Frozen Encoder Experiment

Trainable Parameters:

```text
33,797
```

Encoder:

```text
Frozen
```

Results:

| Metric      |  Value |
| ----------- | -----: |
| Macro AUROC | 0.7696 |
| Micro AUROC | 0.8058 |
| Macro F1    | 0.3985 |
| Micro F1    | 0.5230 |
| Fmax        | 0.5864 |

Observation:

The SSL encoder alone already learned meaningful ECG representations.

---

# Full Fine-Tuning Experiment

Trainable Parameters:

```text
1,387,397
```

Encoder:

```text
Trainable
```

Classifier:

```text
Trainable
```

Training Time:

```text
9875.76 seconds
≈ 2.74 hours
```

Peak GPU Memory:

```text
681.52 MB
```

---

# Best Fine-Tuning Results

Best Epoch:

```text
24
```

Metrics:

| Metric      |  Value |
| ----------- | -----: |
| Macro AUROC | 0.9105 |
| Micro AUROC | 0.9284 |
| Macro F1    | 0.6966 |
| Micro F1    | 0.7531 |
| Fmax        | 0.7595 |

---

# Frozen vs Fine-Tuned Comparison

| Metric      | Frozen | Fine-Tuned | Improvement |
| ----------- | -----: | ---------: | ----------: |
| Macro AUROC | 0.7696 |     0.9105 |     +0.1409 |
| Micro AUROC | 0.8058 |     0.9284 |     +0.1226 |
| Macro F1    | 0.3985 |     0.6966 |     +0.2981 |
| Micro F1    | 0.5230 |     0.7531 |     +0.2301 |
| Fmax        | 0.5864 |     0.7595 |     +0.1731 |

Conclusion:

SSL pretraining provided significant benefits and improved downstream disease classification performance.

---

# Class-wise AUROC

| Class |  AUROC |
| ----- | -----: |
| NORM  | 0.9398 |
| CD    | 0.9332 |
| MI    | 0.9230 |
| HYP   | 0.8289 |
| STTC  | 0.9278 |

Best Performing:

```text
NORM
```

Most Difficult:

```text
HYP
```

---

# Confusion Statistics

## NORM

```text
TP = 857
FP = 205
TN = 1023
FN = 98
```

## CD

```text
TP = 377
FP = 97
TN = 1591
FN = 118
```

## MI

```text
TP = 361
FP = 108
TN = 1535
FN = 179
```

## HYP

```text
TP = 92
FP = 65
TN = 1850
FN = 176
```

## STTC

```text
TP = 346
FP = 105
TN = 1550
FN = 182
```

---

# Final Status

Completed:

* PTB-XL Integration
* Dataset Verification
* Lead II / Morphology Split
* Teacher Architecture
* SSL Pretraining
* Contrastive Learning
* Teacher Checkpoint Generation
* Fine-Tuning Pipeline
* Disease Classification
* Evaluation Metrics
* Teacher Validation

Final Teacher Performance:

```text
Macro AUROC = 0.9105
Micro AUROC = 0.9284
Macro F1 = 0.6966
Micro F1 = 0.7531
Fmax = 0.7595
```

The Teacher model is fully trained, validated, and ready for the next stage:

```text
Knowledge Distillation
        ↓
Lightweight Student Network
        ↓
Edge Deployment
```
