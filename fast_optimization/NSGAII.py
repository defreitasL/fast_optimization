import numpy as np
from numba import jit

# @jit(nopython=True)
# def initialize_population(population_size, Y):
#     # Define os limites logarítmicos para cada parâmetro (a, b, cacr, cero, Yini)
#     log_lower_bounds = np.array([np.log(1e-5), np.log(1e-4), np.log(1e-6), np.log(1e-6), Y[0]])
#     log_upper_bounds = np.array([np.log(1e+2), np.log(1e+3), np.log(1e-1), np.log(1e-1), Y[1]])
    
#     # Inicializa a população dentro dos limites logarítmicos
#     population = np.zeros((population_size, 5))
#     for i in range(5):
#         population[:, i] = np.random.uniform(log_lower_bounds[i], log_upper_bounds[i], population_size)
#     return population, log_lower_bounds, log_upper_bounds

# @jit(nopython=True)
# def model_simulation(params, E, dt, idx_obs):
#     # Executa o modelo com parâmetros dados transformados
#     a, b, cacr, cero, Yini = params
#     Ymd, _ = yates09(E, dt, -np.exp(a), np.exp(b), -np.exp(cacr), -np.exp(cero), Yini)
#     return Ymd[idx_obs]

@jit(nopython=True)
def fast_non_dominated_sort(objectives):
    population_size = objectives.shape[0]
    domination_count = np.zeros(population_size, dtype=np.int32)
    dominated_solutions = np.full((population_size, population_size), -1, dtype=np.int32)
    current_counts = np.zeros(population_size, dtype=np.int32)
    ranks = np.zeros(population_size, dtype=np.int32)

    # Array to store fronts as indices
    front_indices = np.full((population_size, population_size), -1, dtype=np.int32)
    front_sizes = np.zeros(population_size, dtype=np.int32)

    for p in range(population_size):
        for q in range(population_size):
            if np.all(objectives[p] <= objectives[q]) and np.any(objectives[p] < objectives[q]):
                dominated_solutions[p, current_counts[p]] = q
                current_counts[p] += 1
            elif np.all(objectives[q] <= objectives[p]) and np.any(objectives[q] < objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            ranks[p] = 0
            front_indices[0, front_sizes[0]] = p
            front_sizes[0] += 1

    i = 0
    while front_sizes[i] > 0:
        next_front_size = 0
        for j in range(front_sizes[i]):
            p = front_indices[i, j]
            for k in range(current_counts[p]):
                q = dominated_solutions[p, k]
                if q == -1:
                    break
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    ranks[q] = i + 1
                    front_indices[i+1, next_front_size] = q
                    next_front_size += 1
        front_sizes[i+1] = next_front_size
        i += 1

    return ranks, front_indices, front_sizes

@jit(nopython=True)
def calculate_crowding_distance(objectives, front):
    num_individuals = len(front)
    num_objectives = objectives.shape[1]
    distance = np.zeros(num_individuals)

    # Iterar sobre cada objetivo para calcular a distância de crowding
    for m in range(num_objectives):
        sorted_indices = np.argsort(objectives[front, m])  # Ordenar indivíduos por este objetivo
        distance[sorted_indices[0]] = distance[sorted_indices[-1]] = np.inf  # Atribuir infinito para os extremos

        # Calcula as distâncias de crowding para cada indivíduo, exceto os extremos
        for i in range(1, num_individuals - 1):
            next_value = objectives[front[sorted_indices[i + 1]], m]
            prev_value = objectives[front[sorted_indices[i - 1]], m]
            norm = objectives[front[sorted_indices[-1]], m] - objectives[front[sorted_indices[0]], m]
            if norm != 0:
                distance[sorted_indices[i]] += (next_value - prev_value) / norm

    return distance

@jit(nopython=True)
def fast_sort(scores):
    num_individuals = scores.shape[0]
    ranks = np.zeros(num_individuals, dtype=np.int32)
    
    for i in range(num_individuals):
        for j in range(num_individuals):
            if np.all(scores[i] <= scores[j]) and np.any(scores[i] < scores[j]):
                ranks[j] += 1
    
    return ranks

@jit(nopython=True)
def select_next_generation(population, scores, population_size):
    ranks = fast_sort(scores)
    return population[np.argsort(ranks)[:population_size]]

@jit(nopython=True)
def tournament_selection(scores, pressure):
    n_select = len(scores)
    n_random = n_select * pressure
    indices = np.random.permutation(np.arange(n_select).repeat(pressure))[:n_random]
    indices = indices.reshape(n_select, pressure)

    selected_indices = np.empty(n_select, dtype=np.int32)
    for i in range(n_select):
        selected_indices[i] = indices[i, np.argmin(scores[indices[i]])]
    return selected_indices

@jit(nopython=True)
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    for i in range(0, len(population), 2):
        if i + 1 >= len(population):
            break
        if np.random.rand() < crossover_prob:
            cross_point = np.random.randint(1, num_vars)
            # Realiza o crossover
            temp = population[i, cross_point:].copy()
            population[i, cross_point:] = population[i + 1, cross_point:].copy()
            population[i + 1, cross_point:] = temp

            # Aplica a verificação de limites para garantir que os valores estejam dentro dos limites
            for j in range(cross_point, num_vars):
                population[i, j] = min(max(population[i, j], lower_bounds[j]), upper_bounds[j])
                population[i + 1, j] = min(max(population[i + 1, j], lower_bounds[j]), upper_bounds[j])
    
    return population

@jit(nopython=True)
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds):
    for i in range(len(population)):
        for j in range(num_vars):
            if np.random.rand() < mutation_rate:
                # A mutação é um ajuste pequeno de 5% da faixa do parâmetro
                range_val = upper_bounds[j] - lower_bounds[j]
                delta = np.random.uniform(-0.05 * range_val, 0.05 * range_val)
                mutated_value = population[i, j] + delta
                # Garante que o valor mutado esteja dentro dos limites
                mutated_value = max(lower_bounds[j], min(mutated_value, upper_bounds[j]))
                population[i, j] = mutated_value
    return population

@jit(nopython=True)
def fitness_sharing(objectives, sigma_share=0.01):
    population_size = len(objectives)
    distances = np.zeros((population_size, population_size))
    sharing_values = np.zeros((population_size, population_size))

    # Calcula a distância euclidiana entre cada par de soluções
    for i in range(population_size):
        for j in range(i + 1, population_size):
            distances[i, j] = distances[j, i] = np.sqrt(np.sum((objectives[i] - objectives[j])**2))

    # Calcula os valores de compartilhamento
    for i in range(population_size):
        for j in range(population_size):
            if distances[i, j] < sigma_share:
                sharing_values[i, j] = 1 - (distances[i, j] / sigma_share)
            else:
                sharing_values[i, j] = 0

    # Aplica o compartilhamento de fitness
    niche_counts = np.sum(sharing_values, axis=1)
    modified_fitness = np.zeros(population_size)
    for i in range(population_size):
        modified_fitness[i] = 1.0 / (1.0 + niche_counts[i])

    return modified_fitness

@jit(nopython=True)
def select_niched_population(population, objectives, num_to_select):
    modified_fitness = fitness_sharing(objectives)
    selected_indices = np.argsort(-modified_fitness)[:num_to_select]
    return population[selected_indices], objectives[selected_indices]

@jit(nopython=True)
def nsgaii_algorithm(objective_function, model_simulation, Obs, initialize_population, num_generations, population_size, cross_prob, mutation_rate, regeneration_rate):
    # Inicializar a população
    population, lower_bounds, upper_bounds = initialize_population(population_size)
    npar = population.shape[1]
    objectives = np.empty((population_size, 3))  # Três objetivos: KGE, NSE, MSS

    # Avaliar a população inicial
    for i in range(population_size):
        # simulation = model_simulation(population[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
        simulation = model_simulation(population[i])
        objectives[i] = objective_function(Obs, simulation)

    # Proporção da população a ser regenerada a cada geração
    regeneration_rate = 0.1  # Por exemplo, 10% da população
    num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

    # Loop principal do algoritmo NSGA-II
    for generation in range(num_generations):
        ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
        next_population_indices = np.empty(0, dtype=np.int32)

        current_size = 0
        i = 0
        while i < population_size and front_sizes[i] > 0:
            current_front = front_indices[i, :front_sizes[i]]
            crowding_distances = calculate_crowding_distance(objectives, current_front)
            sorted_indices = np.argsort(-crowding_distances)  # Ordena descendente
            selected_indices = current_front[sorted_indices]

            if current_size + len(selected_indices) > population_size - num_to_regenerate:
                remaining_space = population_size - num_to_regenerate - current_size
                next_population_indices = np.append(next_population_indices, selected_indices[:remaining_space])
                break
            next_population_indices = np.append(next_population_indices, selected_indices)
            current_size += len(selected_indices)
            i += 1

        # Criar o pool de acasalamento e gerar a próxima geração
        mating_pool = population[next_population_indices.astype(np.int32)]
        offspring = crossover(mating_pool, npar, cross_prob, lower_bounds, upper_bounds)
        offspring = polynomial_mutation(offspring, mutation_rate, npar, lower_bounds, upper_bounds)

        # Reintroduzir novos indivíduos aleatórios para manter a diversidade genética
        new_individuals, _, _ = initialize_population(num_to_regenerate)
        offspring = np.vstack((offspring, new_individuals))

        # Avaliar a nova população de filhos
        new_objectives = np.empty_like(objectives)
        for i in range(population_size):
            # simulation = model_simulation(offspring[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
            simulation = model_simulation(offspring[i])
            new_objectives[i] = objective_function(Obs, simulation)

        # Preparar a próxima geração
        population = offspring
        objectives = new_objectives

    return population, objectives

@jit(nopython=True)
def nsgaii_algorithm_ts(objective_function, model_simulation, Obs, initialize_population, num_generations, population_size, pressure, regeneration_rate):

    # Inicializar a população
    population, lower_bounds, upper_bounds = initialize_population(population_size)
    npar = population.shape[1]
    objectives = np.empty((population_size, 3))  # Três objetivos: KGE, NSE, MSS

    # Avaliar a população inicial
    for i in range(population_size):
        # simulation = model_simulation(population[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
        simulation = model_simulation(population[i])
        objectives[i] = objective_function(Obs, simulation)

    # Proporção da população a ser regenerada a cada geração
    num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

    # Loop principal do algoritmo NSGA-II
    for generation in range(num_generations):
        ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
        # ranks = fast_sort(objectives)
        next_population_indices = tournament_selection(ranks, pressure)  # Seleção por torneio com pressão

        # Criar o pool de acasalamento e gerar a próxima geração
        mating_pool = population[next_population_indices.astype(np.int32)]
        offspring = crossover(mating_pool, npar, 0.9, lower_bounds, upper_bounds)
        offspring = polynomial_mutation(offspring, 0.1, npar, lower_bounds, upper_bounds)

        # Reintroduzir novos indivíduos aleatórios para manter a diversidade genética
        new_individuals, _, _ = initialize_population(num_to_regenerate)
        offspring = np.vstack((offspring, new_individuals))

        # Avaliar a nova população de filhos
        new_objectives = np.empty_like(objectives)
        for i in range(population_size):
            # simulation = model_simulation(offspring[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
            simulation = model_simulation(offspring[i])
            new_objectives[i] = objective_function(Obs, simulation)

        # Preparar a próxima geração
        population = offspring
        objectives = new_objectives


    return population, objectives
