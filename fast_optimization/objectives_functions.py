from .metrics import *

def multi_obj_indexes(metrics):
    metrics_name_list, _ = backtot()
    print('**Using the following metrics:**')
    idx = []
    for metric in metrics:
        idx.append(metrics_name_list.index(metric)) 
    for i in idx:
        print(metrics_name_list[i])

    return idx

# @jit
def multi_obj_func(evaluation, simulation, indexes):
    _, mask = backtot()
    likes = []
    for i in indexes:
        if mask[i]:
            likes.append(opt(i, evaluation, simulation))
        else:
            likes.append(1 - opt(i, evaluation, simulation))
    
    return likes

def calculate_metrics(evaluation, simulation, indexes):
    metrics_names, _ = backtot()
    likes = []
    for i in indexes:
        likes.append(opt(i, evaluation, simulation))
    
    return likes, metrics_names

def select_best_solution(objectives):
    # Normalizar os objetivos
    min_values = np.min(objectives, axis=0)
    max_values = np.max(objectives, axis=0)
    normalized_objectives = (objectives - min_values) / (max_values - min_values)

    # Calcular a soma ponderada (pesos iguais para todos os objetivos)
    weights = np.ones(objectives.shape[1]) / objectives.shape[1]
    weighted_sum = np.dot(normalized_objectives, weights)

    # Selecionar a solução com o menor valor da soma ponderada
    best_index = np.argmin(weighted_sum)
    
    return best_index, weighted_sum[best_index]
