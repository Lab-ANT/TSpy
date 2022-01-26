# Created by Chengyu on 2022/1/26.

from sklearn import metrics
import numpy as np
from TSpy.label import reorder_label

def __adjusted_macro_score(groundtruth, prediction, score_type):
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
        return

    length = len(groundtruth)

    # convert to numpy array.
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)

    groundtruth_set = set(groundtruth)
    prediction_set = set(prediction)
    used_label_set = set()
    n = len(groundtruth_set)

    total = 0
    for i in groundtruth_set:
        used_j = 0
        max_score = 0
        for j in prediction_set:
            if j in used_label_set:
                continue

            # convert to binary array of positive label.
            true = np.zeros(length, dtype=int)
            pred = np.zeros(length, dtype=int)
            true[np.argwhere(groundtruth == i)]=1
            pred[np.argwhere(prediction == j)]=1

            if score_type == 'f1':
                score = metrics.f1_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'precision':
                score = metrics.precision_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'recall':
                score = metrics.recall_score(true, pred, average='binary', pos_label=1, zero_division=0)
            else:
                print('Error: Score type does not exists.')

            if score > max_score:
                max_score = score
                used_j = j
                
        used_label_set.add(used_j)
        total += max_score
    return total/n

def adjusted_macro_recall(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='recall')

def adjusted_macro_precision(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='precision')

def adjusted_macro_f1score(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='f1')

def macro_recall(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.recall_score(groundtruth, prediction, average='macro', zero_division=0)

def macro_precision(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.precision_score(groundtruth, prediction, average='macro', zero_division=0)

def macro_f1score(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro f1 score
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.f1_score(groundtruth, prediction, average='macro', zero_division=0)

def ARI(prediction, groundtruth):
    return metrics.adjusted_rand_score(groundtruth, prediction)

def AMI(prediction, groundtruth):
    return metrics.adjusted_mutual_info_score(groundtruth, prediction)

def evaluation(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    ami = AMI(groundtruth, prediction)
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall, ari, ami