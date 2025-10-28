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
    metrics_names = [metrics_names[i] for i in indexes]
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

def select_best_from_first_front(objectives_min):  # ya en modo minimizar
    eps = 1e-12
    # normalización por columna (robusta a min==max)
    col_min = objectives_min.min(axis=0)
    col_max = objectives_min.max(axis=0)
    rng = np.maximum(col_max - col_min, eps)
    Z = (objectives_min - col_min) / rng  # 0 = ideal empírico por columna

    # distancia euclídea al punto ideal (0,...,0)
    d = np.linalg.norm(Z, axis=1)
    return np.argmin(d), d.min()

def select_best_solution_L2(objectives):
    """
    Selecciona un único compromiso a partir de objetivos de minimización.
    Usa normalización ideal/nadir y distancia L2 al ideal (0,...,0).
    """
    # ideal y nadir (por columnas)
    ideal = np.min(objectives, axis=0)
    nadir = np.max(objectives, axis=0)

    # evitar división por cero en columnas degeneradas
    denom = np.where(np.abs(nadir - ideal) < 1e-12, 1.0, (nadir - ideal))
    Z = (objectives - ideal) / denom

    # distancia L2 al ideal (0,...,0)
    d = np.sqrt(np.sum(Z**2, axis=1))
    best_index = np.argmin(d)
    return best_index, d[best_index]