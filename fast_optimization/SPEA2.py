import numpy as np
import numba
from .NSGAII import tournament_selection

@numba.jit(nopython=True)
def spea2_algorithm(objective_function, model_simulation, Obs, initialize_population, num_generations, population_size, pressure, regeneration_rate, cross_prob, mutation_rate, mutation_variance, ):
    # Inicialização da população
    y_min_max = np.array([min(Obs), max(Obs)])
    population, lb, ub = initialize_population(population_size, y_min_max)
    npar = population.shape[1]
    objectives = np.zeros((population_size, 3))

    # Avaliação inicial da população
    for i in range(population_size):
        # simulation = model_simulation(population[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
        simulation = model_simulation(population[i])
        objectives[i] = objective_function(Obs, simulation)
   
    # Proporção da população a ser regenerada a cada geração
    num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

    # Loop principal do algoritmo SPEA2
    for generation in range(num_generations):
        # Seleção ambiental e geração de arquivos
        # population, objectives = remove_identical_solutions(population, objectives)
        archive, archive_fitness, archive_objectives = environmental_selection(population, objectives)
        # Seleção dos pais e reprodução
        # mating_pool = tournament_selection2(archive, archive_fitness)
        pool_indexes = tournament_selection(archive_fitness, pressure)
        mating_pool = archive[pool_indexes]
        offspring = genetic_operators2(mating_pool, cross_prob, mutation_rate, npar, lb, ub, mutation_variance)

        # Avaliação da prole
        offspring_objectives = np.zeros((len(offspring),3))
        for i in range(len(offspring)):
            # simulation = model_simulation(offspring[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
            simulation = model_simulation(offspring[i])
            offspring_objectives[i] = objective_function(Obs, simulation)

        # Reintroduzir novos indivíduos aleatórios para manter a diversidade genética
        new_individuals, _, _ = initialize_population(num_to_regenerate, y_min_max)
        new_individuals_objectives = np.zeros((num_to_regenerate, 3))
        for i in range(num_to_regenerate):
            # simulation = model_simulation(new_individuals[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
            simulation = model_simulation(new_individuals[i])
            new_individuals_objectives[i] = objective_function(Obs, simulation)
        
        population = np.vstack((archive, offspring, new_individuals))
        objectives = np.vstack((archive_objectives, offspring_objectives, new_individuals_objectives))
        
        if len(population) < population_size:
            additional_pop, _, _= initialize_population(population_size - len(population), y_min_max)
            additional_obj = np.zeros((population_size - len(population), 3))
            for i in range(len(additional_pop)):
                # simulation = model_simulation(additional_pop[i], E_splited, dt, idx_obs_splited, Obs_splited[0])
                simulation = model_simulation(additional_pop[i])
                additional_obj[i] = objective_function(Obs, simulation)
            population = np.vstack((population, additional_pop))
            objectives = np.vstack((objectives, additional_obj))

    return population, objectives

@numba.jit(nopython=True)
def environmental_selection(population, objectives):

    npop = len(population)
    dist = euclidean_distances(objectives)

    m = 2  # Assuming m=2
    # for d in range(len(dist)):
    #     for i in range(len(dist[d])):
    #         if dist[d][i] == 0:
    #             dist[d][i] = 1e-10
    # idx_zero = np.where((dist == 0))[0]
    # dist[idx_zero] = 1e-10
    k = np.mean(dist) / 5
    if k == 0:
        k = 1e-6
    # Calculate raw fitness and density estimates
    fitness_values = np.sum(np.exp(-1 * (dist ** 2) / (2 * k ** 2)), axis=1)
    density_estimate = np.zeros(npop)

    for i in range(npop):
        sorted_distances = np.sort(dist[i, :])
        if sorted_distances[m+1] == 0:
            sorted_distances[m+1] = 1e-10
        density_estimate[i] = 1 / (2 * k) * (1 / sorted_distances[m+1])  # assuming m=2

    combined_fitness = fitness_values + density_estimate
    median_fitness = np.median(combined_fitness)
    selection_mask = combined_fitness <= median_fitness
    archive = population[selection_mask, :]
    archive_obj = objectives[selection_mask, :]
    archive_fitness = combined_fitness[selection_mask]

    return archive, archive_fitness, archive_obj

@numba.jit(nopython=True)
def tournament_selection2(population, fitness):
    npop, npar = population.shape
    selected = np.empty((npop, npar), dtype=population.dtype)
    indices = random_permutation(npop)

    for i in range(npop):
        # Evitar o problema com o if in-line definindo explicitamente os índices dos competidores
        if i + 1 < npop:
            competitors = [indices[i], indices[i + 1]]
        else:
            # Garantir que sempre temos dois competidores (pode repetir o último se necessário)
            competitors = [indices[i], indices[0]]  # O segundo competidor é o primeiro do array se estivermos no fim

        # Comparar os fitness dos competidores
        if fitness[competitors[0]] < fitness[competitors[1]]:
            selected[i, :] = population[competitors[0], :]
        else:
            selected[i, :] = population[competitors[1], :]

    return selected

@numba.jit(nopython=True)
def genetic_operators2(parents, crossover_prob, mutation_rate, num_vars, lower_bounds, upper_bounds, mutation_variance=0.05):
    num_parents, npar = parents.shape
    num_offspring = num_parents - (num_parents % 2)  # Garante um número par de filhos
    offspring = np.empty((num_offspring, npar), dtype=parents.dtype)

    # Preparando pais para crossover e mutação
    prepared_parents = np.empty_like(parents)
    for i in range(0, num_parents, 2):
        if i + 1 < num_parents:
            prepared_parents[i:i+2] = crossover(parents[i:i+2], num_vars, crossover_prob, lower_bounds, upper_bounds)

    # Aplica mutação em todos os filhos gerados pelo crossover
    for i in range(num_offspring):
        offspring[i] = polynomial_mutation(prepared_parents[i], mutation_rate, num_vars, lower_bounds, upper_bounds, mutation_variance)

    return offspring

@numba.jit(nopython=True)
def crossover(parents, num_vars, crossover_prob, lower_bounds, upper_bounds):
    offspring = parents.copy()
    if np.random.rand() < crossover_prob:
        cross_point = np.random.randint(1, num_vars)
        # Realiza o crossover
        offspring[0, cross_point:], offspring[1, cross_point:] = offspring[1, cross_point:].copy(), offspring[0, cross_point:].copy()

        # Aplica a verificação de limites para garantir que os valores estejam dentro dos limites
        # for j in range(cross_point, num_vars):
        #     offspring[0, j] = min(max(offspring[0, j], lower_bounds[j]), upper_bounds[j])
        #     offspring[1, j] = min(max(offspring[1, j], lower_bounds[j]), upper_bounds[j])
    
    return offspring

@numba.jit(nopython=True)
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, mutation_variance=0.05):
    for j in range(num_vars):
        if np.random.rand() < mutation_rate:
            # A mutação é um ajuste pequeno de 5% da faixa do parâmetro
            range_val = upper_bounds[j] - lower_bounds[j]
            delta = np.random.uniform(-mutation_variance * range_val, mutation_variance * range_val)
            mutated_value = population[j] + delta
            # Garante que o valor mutado esteja dentro dos limites
            mutated_value = max(lower_bounds[j], min(mutated_value, upper_bounds[j]))
            population[j] = mutated_value
    return population

@numba.jit(nopython=True)
def euclidean_distances(X):
    # Utilizando broadcasting para calcular a matriz de distâncias de forma vetorizada
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)  # Evitar divisão por zero para a própria distância
    return dist

@numba.jit(nopython=True)
def random_permutation(n):
    indices = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices

@numba.jit(nopython=True)
def remove_identical_solutions(population, objectives):
    n = population.shape[0]
    keep = np.ones(n, dtype=numba.boolean)  # Array para marcar elementos únicos
    
    # Comparar cada par de soluções
    for i in range(n):
        for j in range(i + 1, n):
            if keep[j] and np.all(population[i] == population[j]):
                keep[j] = False  # Marcar como falso se for duplicado

    # Filtrar e retornar apenas as soluções únicas
    return population[keep], objectives[keep]

@numba.jit(nopython=True)
def euclidean_distances2(X):
    first_element = X[0]
    diff = X - first_element
    dist = np.sqrt(np.sum(diff**2, axis=1))
    return dist