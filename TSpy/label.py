import numpy as np

def compact(series):
    '''
    Compact Time Series.
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted

def remove_duplication(series):
    '''
    Remove duplication.
    '''
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result
    
def seg_to_label(label):
    pre = 0
    seg = []
    for l in label:
        seg.append(np.ones(l-pre,dtype=np.int)*label[l])
        pre = l
    result = np.concatenate(seg)
    return result

def adjust_label(label):
    '''
    Adjust label order.
    '''
    label = np.array(label)
    compacted_label = compact(label)
    ordered_label_set = remove_duplication(compacted_label)
    label_set = set(label)
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for idx, elem in zip(idx_list,label_set):
        label[idx] = elem
    return label