import numpy as np
import warnings
import torch
import sklearn.metrics as skm

def original_multi_threshold_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: np.ndarray
):
    """IMLE-Net helper for computing precision and recall across multiple thresholds."""
    # Expand analysis to number of thresholds
    y_pred_bin = (
        np.repeat(y_pred[None, :, :], len(thresholds), axis=0)
        >= thresholds[:, None, None]
    )
    y_true_bin = np.repeat(y_true[None, :, :], len(thresholds), axis=0)

    # Compute true positives
    TP = np.sum(np.logical_and(y_true, y_pred_bin), axis=2)

    # Compute macro-average precision handling all warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sum(y_pred_bin, axis=2)
        precision = TP / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # Compute macro-average recall handling possible zero-label rows safely
    with np.errstate(divide="ignore", invalid="ignore"):
        true_sum = np.sum(y_true_bin, axis=2)
        recall = TP / true_sum
        recall[true_sum == 0] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_recall = np.nanmean(recall, axis=1)

    return av_precision, av_recall


def original_metric_summary(
    y_true: np.ndarray, y_pred: np.ndarray, num_thresholds: int = 10
):
    """IMLE-Net Original Fmax, Precision, and Recall calculation."""
    thresholds = np.arange(0.00, 1.01, 1.0 / (num_thresholds - 1), float)
    average_precisions, average_recalls = original_multi_threshold_precision_recall(
        y_true, y_pred, thresholds
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = (
            2
            * (average_precisions * average_recalls)
            / (average_precisions + average_recalls)
        )
        f_scores[np.isnan(f_scores)] = 0.0
    
    # We find the index of max F1 score
    max_idx = np.nanargmax(f_scores)
    
    return (
        float(f_scores[max_idx]),          # Fmax
        float(average_precisions[max_idx]),# Precision
        float(average_recalls[max_idx]),   # Recall
        float(thresholds[max_idx])         # Optimal Threshold
    )

def compute_multilabel_metrics(y_true, y_pred_probs, data_dir=None):
    """
    Computes multi-label metrics:
    - Macro AUROC
    - Micro AUROC
    - Macro F1 (at default threshold 0.5)
    - Micro F1 (at default threshold 0.5)
    - Fmax (maximum F1 score across thresholds 0.0 to 1.0)
    - cm_score (PhysioNet Challenge Metric) if data_dir is provided
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.cpu().numpy()
        y_true_tensor = y_true
    else:
        y_true_np = np.array(y_true)
        y_true_tensor = torch.tensor(y_true)

    if isinstance(y_pred_probs, torch.Tensor):
        y_prob_np = y_pred_probs.cpu().numpy()
        y_prob_tensor = y_pred_probs
    else:
        y_prob_np = np.array(y_pred_probs)
        y_prob_tensor = torch.tensor(y_pred_probs)
        
    y_true_np = y_true_np.astype(int)

    # Avoid NaNs if single class happens to have no positive samples
    try:
        macro_auroc = skm.roc_auc_score(y_true_np, y_prob_np, average='macro')
    except ValueError:
        macro_auroc = 0.5

    try:
        micro_auroc = skm.roc_auc_score(y_true_np, y_prob_np, average='micro')
    except ValueError:
        micro_auroc = 0.5

    # F1 scores at default 0.5 threshold
    y_pred_bin = (y_prob_np >= 0.5).astype(int)
    macro_f1 = skm.f1_score(y_true_np, y_pred_bin, average='macro', zero_division=0)
    micro_f1 = skm.f1_score(y_true_np, y_pred_bin, average='micro', zero_division=0)

    imle_thresholds = np.arange(0.01, 1.0, 0.01)
    imle_best_f1_macro = 0.0
    imle_best_threshold = 0.5

    for t in imle_thresholds:
        y_pred_t = (y_prob_np >= t).astype(int)
        f1_t = skm.f1_score(y_true_np, y_pred_t, average='macro', zero_division=0)
        if f1_t > imle_best_f1_macro:
            imle_best_f1_macro = f1_t
            imle_best_threshold = t

    orig_fmax, orig_prec, orig_rec, orig_thresh = original_metric_summary(y_true_np, y_prob_np)
    
    cm_score = 0.0
    if data_dir is not None:
        try:
            cm_score = cal_Acc(y_true_tensor, y_prob_tensor, data_dir=data_dir)
        except Exception as e:
            print("CM Score Calculation Failed:", e)

    return {
        "macro_auroc": float(macro_auroc),
        "micro_auroc": float(micro_auroc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "fmax": float(imle_best_f1_macro), # Retained for backward compatibility
        "IMLE_Macro_Fmax": float(imle_best_f1_macro),
        "IMLE_Original_Fmax": float(orig_fmax),
        "cm_score": float(cm_score)
    }

import pandas as pd
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_table(table_file):
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    num_rows = len(table)-1
    if num_rows < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols) != 1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

def load_weights(weight_file, classes):
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]

    return weights

def compute_modified_confusion_matrix(labels, outputs):
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    for i in range(num_recordings):
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        for j in range(num_classes):
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

def compute_challenge_metric(weights, labels, outputs, classes, normal_class):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal_class)

    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

def cal_Acc(y_true, y_pre, data_dir=None, threshold=0.5, num_classes=24, beta=2, normal=False):
    import os
    import pandas as pd
    import numpy as np
    import torch
    
    y_true = y_true.cpu().detach().numpy().astype(int)

    y_label = np.zeros(y_true.shape)
    _, y_pre_label = torch.max(y_pre, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(y_true.shape[0]), y_pre_label] = 1
    y_pre = y_pre.cpu().detach().numpy() >= threshold

    y_label = y_label + y_pre
    y_label[y_label > 1.1] = 1

    labels = y_true
    binary_outputs = y_label

    if data_dir is None: return 0.0
    dx_mapping_path = os.path.join(data_dir, 'dx_mapping_scored.csv')
    weights_file = os.path.join(data_dir, 'weights.csv')
    normal_class = '426783006'

    label_file = pd.read_csv(dx_mapping_path)
    equivalent_classes = ['59118001', '63593006', '17338001']
    classes = sorted(list(set([str(name) for name in label_file['SNOMED CT Code']]) - set(equivalent_classes)))

    weights = load_weights(weights_file, classes)

    indices = np.any(weights, axis=0)
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]

    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    return challenge_metric

