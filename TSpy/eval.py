from sklearn import metrics
import numpy as np
from TSpy.label import reorder_label

def evaluation(groundtruth, prediction):
    pass

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
