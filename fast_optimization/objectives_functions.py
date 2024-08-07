from .metrics import *
from numba import jit

# @jit(nopython=True)
def objective_functions(metric):
    
    index_metric = metrics_name_list.index(metric)
    index_metric.append(metrics_name_list.index(metric))

    if mask[index_metric]:
        @jit(nopython=True)
        def obj_func(evaluation, simulation):
            return opt[index_metric](evaluation, simulation)
    else:
        @jit(nopython=True)
        def obj_func(evaluation, simulation):
            return 1 - opt[index_metric](evaluation, simulation)

    return obj_func

# @jit(nopython=True)
def multi_obj_func(metrics):
    print('**Using the following metrics:**')
    idx = []
    for metric in metrics:
        idx.append(metrics_name_list.index(metric)) 
    for i in idx:
        print(metrics_name_list[i])

    @jit(nopython=True)
    def obj_func(evaluation, simulation):
        likes = []
        for i in idx:
            if mask[i]:
                likes.append(opt[i](evaluation, simulation))
            else:
                likes.append(1 - opt[i](evaluation, simulation))
        
        return likes
          
    return obj_func 
