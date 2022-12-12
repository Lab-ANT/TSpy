import os
from TSpy.label import seg_to_label
import scipy.io
import numpy as np
import json

def load_UEA(dataset, path):
    path = os.path.join(path, dataset+'/'+dataset)
    train_data = np.load(path+'_TRAIN.npy')
    test_data = np.load(path+'_TEST.npy')
    train_labels = np.load(path+'_TRAIN_labels.npy')
    test_labels = np.load(path+'_TEST_labels.npy')
    with open(path+'_map.json') as f:
        label_map = json.load(f)
        f.close()
    return train_data, train_labels, test_data, test_labels, label_map

def load_general_seg_dagaset(path):
    data = np.load(path)
    mts = data[:,:-1]
    label = data[:,-1]
    # print(mts.shape, label.shape)
    return mts, label

def load_USC_HAD(subject, target, dataset_path):
    prefix = os.path.join(dataset_path,'USC-HAD/Subject'+str(subject)+'/')
    fname_prefix = 'a'
    fname_postfix = 't'+str(target)+'.mat'
    data_list = []
    label_json = {}
    total_length = 0
    for i in range(1,13):
        data = scipy.io.loadmat(prefix+fname_prefix+str(i)+fname_postfix)
        data = data['sensor_readings']
        data_list.append(data)
        total_length += len(data)
        label_json[total_length]=i
    label = seg_to_label(label_json)
    return np.vstack(data_list), label