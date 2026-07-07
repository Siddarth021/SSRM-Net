import numpy as np
import pandas as pd
import torch
import copy
import sklearn.metrics as skm

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


def cal_Acc(y_true, y_pre, threshold=0.5, num_classes=9, beta=2, normal=False, weights_file='/scratch/physionet-2020/weights.csv', label_file_dir='/scratch/physionet-2020/dx_mapping_scored.csv'):
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

    normal_class = '426783006'
    label_file = pd.read_csv(label_file_dir)
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


def get_predictions(labels, predictions, threshold):
    labels_numpy = labels.cpu().detach().numpy().astype(int)
    y_label = np.zeros(labels_numpy.shape)

    _, y_pre_label = torch.max(predictions, 1)
    y_pre_label = y_pre_label.cpu().detach().numpy()

    y_label[np.arange(labels_numpy.shape[0]), y_pre_label] = 1
    y_pre = predictions.cpu().detach().numpy() >= threshold

    y_predictions = y_pre + y_label
    y_predictions[y_predictions > 1.1] = 1
    
    return y_predictions
