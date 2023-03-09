'''
Created by Chengyu on 2022/2/7.
'''

from sklearn import metrics
import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd

def partial_state_correlation(X,Y):
    length = len(X)
    set_X = list(set(X))
    set_Y = list(set(Y))
    matrix = np.zeros((len(set_X),len(set_Y)))
    for s_x in set_X:
        for s_y in set_Y:
            X_ = -np.ones(length, dtype=int)
            X_[np.argwhere(X==s_x)] = 1
            X_[np.argwhere(X!=s_x)] = 0
            Y_ = -np.ones(length, dtype=int)
            Y_[np.argwhere(Y==s_y)] = 1
            Y_[np.argwhere(Y!=s_y)] = 0
            # print(X_, np.argwhere(X==s_x))
            # print(X_, Y_)
            # NMI_score = metrics.normalized_mutual_info_score(X_, Y_)
            NMI_score = metrics.accuracy_score(X_,Y_)
            matrix[s_x, s_y] = NMI_score
    return matrix

# data1 = np.array([0,0,0,0,1,1,2,2,3,3,3,3])
# data2 = np.array([0,0,1,1,1,1,2,2,2,2,2,2])

def match_label(X, Y):
    matrix = partial_state_correlation(X, Y)
    # print(matrix)
    set_X = list(set(X))
    set_Y = list(set(Y))
    row_len = len(set_Y)
    adjust_list = []
    while len(set_X) != 0 and len(set_Y) != 0:
        pos = np.argmax(matrix)
        i = int(pos/row_len)
        j = int(pos%row_len)
        # print(pos, i,'->',j)
        # print(matrix.shape)
        matrix[i,:] = 0
        matrix[:,j] = 0
        set_X.remove(i)
        set_Y.remove(j)
        adjust_list.append((i,j))
    set_Y = list(set(Y))
    for pair in adjust_list:
        symbol = pair[1]
        Y[np.argwhere(Y==pair[1])] = pair[0]+5
    return Y

# def lagged_NMI(seq1, seq2, ratio):
#     atom_step = 0.01
#     length = len(seq1)
#     k = int(ratio/atom_step)
#     max_score = -1
#     lag = 0
#     for i in range(k):
#         lag_len = int(k*atom_step*length)
#         seq1 = np.concatenate([np.zeros(lag_len).flatten(), seq1[lag_len:].flatten()])
#         NMI_score = metrics.normalized_mutual_info_score(seq1,seq2)
#         if NMI_score >= max_score:
#             max_score = NMI_score
#             lag = lag_len
#     return max_score, lag

# def lagged_NMI(seq1, seq2, ratio, atom_step=0.05):
#     length = len(seq1)
#     k = int(ratio/atom_step)
#     max_score = -1
#     lag = 0
#     for i in range(-k, k+1):
#         lag_len = int(i*atom_step*length)
#         if lag_len>0:
#             seq = np.concatenate([np.zeros(lag_len).flatten(), seq1[lag_len:].flatten()])
#         elif lag_len<0:
#             seq = np.concatenate([seq1[:lag_len].flatten(), np.zeros(-lag_len).flatten()])
#         else:
#             pass
#         NMI_score = metrics.normalized_mutual_info_score(seq,seq2)
#         if NMI_score >= max_score:
#             max_score = NMI_score
#             lag = lag_len
#     return max_score, lag

def lagged_NMI(seq1, seq2, ratio, atom_step=0.001):
    length = len(seq1)
    k = int(ratio/atom_step)
    max_score = -1
    lag = 0
    for i in range(-k, k+1):
        lag_len = int(i*atom_step*length)
        if lag_len>0:
            NMI_score = metrics.normalized_mutual_info_score(seq1[lag_len:],seq2[:-lag_len])
        elif lag_len<0:
            NMI_score = metrics.normalized_mutual_info_score(seq1[:lag_len],seq2[-lag_len:])
        else:
            NMI_score = metrics.normalized_mutual_info_score(seq1,seq2)
        # print(i, lag_len, NMI_score)
        if NMI_score >= max_score:
            max_score = NMI_score
            lag = lag_len
    return max_score, lag

def lagged_state_correlation(seq_list, ratio=0.05):
    '''
    Lagged state correlation.
    @Params:
        seq_list: list of state sequences.
        ratio: maximum lag ratio.
    @return:
        correlation_matrix: state correlation matrix.
        lag_matrix: lag matrix.
    '''
    num_instance = len(seq_list)
    correlation_matrix = np.ones((num_instance,num_instance))
    lag_matrix = np.ones((num_instance, num_instance))
    for i in range(num_instance):
        for j in range(num_instance):
            if i < j:
                NMI_score, lag = lagged_NMI(seq_list[i],seq_list[j], ratio)
                correlation_matrix[i,j] = NMI_score
                correlation_matrix[j,i] = NMI_score
                lag_matrix[i,j] = lag
                lag_matrix[j,i] = -lag
            else:
                continue
    return correlation_matrix, lag_matrix

def state_correlation(seq_list):
    length = len(seq_list)
    correlation_matrix = np.ones((length,length))
    for i in range(length):
        for j in range(length):
            if i < j:
                NMI_score = metrics.normalized_mutual_info_score(seq_list[i],seq_list[j])
                correlation_matrix[i,j] = NMI_score
                correlation_matrix[j,i] = NMI_score
            else:
                continue
    return correlation_matrix

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_to_cluster_array, idx