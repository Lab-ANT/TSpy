from sklearn import metrics
import numpy as np
from TSpy.label import adjust_label

def macro_precision(prediction, groundtruth):
    '''
    calculate macro precision
    '''
    prediction = np.array(prediction)
    groundtruth = np.array(groundtruth)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        return metrics.precision_score(adjust_label(prediction),adjust_label(groundtruth),average='macro', zero_division=0)

def macro_f1score(prediction, groundtruth):
    '''
    calculate macro f1 score
    '''
    prediction = np.array(prediction)
    groundtruth = np.array(groundtruth)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        return metrics.f1_score(adjust_label(prediction),adjust_label(groundtruth),average='macro', zero_division=0)

def ARI(prediction, groundtruth):
    return metrics.adjusted_rand_score(prediction, groundtruth)

def AMI(prediction, groundtruth):
    return metrics.adjusted_mutual_info_score(prediction, groundtruth)
