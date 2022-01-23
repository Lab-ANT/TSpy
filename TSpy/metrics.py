import sklearn.metrics as metrics
import numpy as np

def calculate_NMI_matrix(seq_list):
    result = np.zeros(shape=(100,100))
    for i in range(100):
        for j in range(100):
            nmi = metrics.normalized_mutual_info_score(seq_list[i], seq_list[j])
            result[i,j] = nmi
    return result