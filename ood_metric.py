import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import pickle

with open('bpds_Complex_CIFAR10_2.pkl', 'rb') as f1:
    bpds = pickle.load(f1)

with open('bpds_Complex_SVHN_2.pkl', 'rb') as f2:
    bpds_svhn = pickle.load(f2)[:len(bpds)]

def AUROC(id_data, ood_data):
    y_true = [0]*len(id_data)
    ones = [1]*len(ood_data)
    y_true.extend(ones)
    id_data.extend(ood_data)
    y_score = id_data
    y_true = np.array(np.array(y_true))
    y_score = np.array(np.array(y_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    return (metrics.auc(fpr, tpr))

def AUPRC(id_data, ood_data):
    y_true = [0]*len(id_data)
    ones = [1]*len(ood_data)
    y_true.extend(ones)
    id_data.extend(ood_data)
    y_score = id_data
    y_true = np.array(np.array(y_true))
    y_score = np.array(np.array(y_score))
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score, pos_label=1)
    return (metrics.auc(recall, precision))

def FPR80(id_data, ood_data):
    y_true = [0] * len(id_data)
    ones = [1] * len(ood_data)
    y_true.extend(ones)
    id_data.extend(ood_data)
    y_score = id_data
    y_true = np.array(np.array(y_true))
    y_score = np.array(np.array(y_score))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)

    if all(tpr < 0.80):
        # No threshold allows TPR >= 0.8
        return 0
    elif all(tpr >= 0.80):
        # All thresholds allow TPR >= 0.8, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.80]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.80, tpr, fpr)
