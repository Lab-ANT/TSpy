# Created by Chengyu on 2021/12/8.
# TSpy util collection.

import numpy as np
import math

def all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean)/math.sqrt(var)
        i += 1
    return data_tensor