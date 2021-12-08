from sklearn import metrics
from eval import adjust_label
import numpy as np
from label import adjust_label

def macro_precision(prediction, groundtruth):
    '''
    calculate macro precision
    '''
    prediction = np.array(prediction)
    groundtruth = np.array(groundtruth)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        return metrics.precision_score(adjust_label(prediction),adjust_label(groundtruth),average='macro')

def macro_f1score(prediction, groundtruth):
    '''
    calculate macro f1 score
    '''
    prediction = np.array(prediction)
    groundtruth = np.array(groundtruth)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        return metrics.f1_score(adjust_label(prediction),adjust_label(groundtruth),average='macro')