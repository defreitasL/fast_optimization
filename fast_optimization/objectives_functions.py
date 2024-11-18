from .metrics import *
from numba import jit

def multi_obj_indexes(metrics):
    metrics_name_list, _ = backtot()
    print('**Using the following metrics:**')
    idx = []
    for metric in metrics:
        idx.append(metrics_name_list.index(metric)) 
    for i in idx:
        print(metrics_name_list[i])

    return idx

def multi_obj_func(evaluation, simulation, indexes):
    _, mask = backtot()
    likes = []
    for i in indexes:
        if mask[i]:
            likes.append(opt[i](evaluation, simulation))
        else:
            likes.append(1 - opt[i](evaluation, simulation))
    
    return likes


@jit(nopython=True)
def obj_func(evaluation, simulation, indexes):
    _, mask = backtot()
    likes = np.zeros(len(indexes))
    for i, idx in enumerate(indexes):
        if mask[idx]:
            likes[i]= opt(idx, evaluation, simulation)
        else:
            likes[i]= 1 - opt(idx, evaluation, simulation)
    return likes
