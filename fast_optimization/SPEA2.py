import numpy as np
from numba import jit
import math
from .objectives_functions import multi_obj_func, select_best_solution
from .metrics import backtot

def spea2_algorithm(model_simulation, Obs, initialize_population, num_generations, population_size, cross_prob, mutation_rate, pressure, regeneration_rate, m, eta_mut, kstop, pcento, peps, index_metrics, n_restarts = 5):
    """
    SPEA2 Optimization Algorithm.
    This algorithm aims to optimize an objective function by evolving a population using environmental selection and genetic operations.
    
    Parameters:
    - model_simulation: Function that simulates the model given an input vector of parameters.
    - Obs: Observed data used for calculating the fitness.
    - initialize_population: Function to initialize the population.
    - num_generations: Number of generations to evolve.
    - population_size: Size of the population.
    - cross_prob: Probability of crossover.
    - mutation_rate: Probability of mutation.
    - pressure: Selection pressure for tournament selection.
    - regeneration_rate: Proportion of the population to be regenerated each generation.
    - m: Number of neighbors to consider in environmental selection.
    - eta_mut: Mutation parameter for polynomial mutation.
    - kstop: Number of past evolution loops to assess convergence.
    - pcento: Percentage improvement allowed in the past kstop loops for convergence.
    - peps: Threshold for the normalized geometric range of parameters to determine convergence.
    - index_metrics: Index used for multi-objective evaluation.

    Returns:
    - best_individual: The best solution found.
    - best_fitness: Fitness value of the best solution.
    - best_fitness_history: History of the best fitness values over generations.
    """
    print('Precompilation done!')
    print(f'Starting SPEA2 algorithm with {n_restarts} restarts...')

    metrics_name_list, mask = backtot()
    metrics_name_list = [metrics_name_list[k] for k in index_metrics]
    mask = [mask[k] for k in index_metrics]

    for restart in range(n_restarts):

        print(f'Starting {restart+1}/{n_restarts}')

        best_fitness_history = []
        best_individuals = []
        # Initialize the population
        population, lb, ub = initialize_population(population_size)
        npar = population.shape[1]
        nobj = len(index_metrics)
        objectives = np.zeros((population_size, nobj))

        # Evaluate initial population
        for i in range(population_size):
            simulation = model_simulation(population[i])
            objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

        # Number of individuals to regenerate each generation
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))


        # Main loop of the SPEA2 algorithm
        for generation in range(num_generations):
            # Environmental selection and archive generation
            archive, archive_fitness, archive_objectives = environmental_selection(population, objectives, m)

            # Prevent empty archive
            if len(archive) == 0:
                sorted_indices = np.argsort(archive_fitness)
                archive = population[sorted_indices[:1], :]
                archive_objectives = objectives[sorted_indices[:1], :]
                archive_fitness = archive_fitness[sorted_indices[:1]]

            # Limit archive size to population size
            if len(archive) > population_size:
                archive_indices = np.argsort(archive_fitness)[:population_size]
                archive = archive[archive_indices]
                archive_objectives = archive_objectives[archive_indices]
                archive_fitness = archive_fitness[archive_indices]
            
            # Parent selection and reproduction
            ranks = np.argsort(archive_fitness)  # Update ranks based on fitness
            crowding_distances = crowd_distance(archive_objectives, ranks)
            pool_indexes = tournament_selection_with_crowding(ranks, crowding_distances, pressure)
            mating_pool = archive[pool_indexes]
            
            # Crossover operation
            min_cross_prob = 0.1  # Set a minimum crossover probability to maintain diversity
            adaptive_cross_prob = max(cross_prob * (1 - generation / num_generations), min_cross_prob)
            offspring = crossover(mating_pool, npar, adaptive_cross_prob, lb, ub)
            
            # Mutation operation
            min_mutation_rate = 0.01  # Set a minimum mutation rate to prevent premature convergence
            adaptive_mutation_rate = max(mutation_rate * (1 - generation / num_generations), min_mutation_rate)
            offspring = polynomial_mutation(offspring, adaptive_mutation_rate, npar, lb, ub, eta_mut)

            # Evaluate offspring
            offspring_objectives = np.zeros((len(offspring), nobj))
            for i in range(len(offspring)):
                simulation = model_simulation(offspring[i])
                offspring_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            # Reintroduce new individuals to maintain genetic diversity
            new_individuals, _, _ = initialize_population(num_to_regenerate)
            new_individuals_objectives = np.zeros((num_to_regenerate, nobj))
            for i in range(num_to_regenerate):
                simulation = model_simulation(new_individuals[i])
                new_individuals_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            population = np.vstack((archive, offspring, new_individuals))
            objectives = np.vstack((archive_objectives, offspring_objectives, new_individuals_objectives))

            # Ensure the population size is correct
            if len(population) < population_size:
                additional_pop, _, _ = initialize_population(population_size - len(population))
                additional_obj = np.zeros((population_size - len(population), nobj))
                for i in range(len(additional_pop)):
                    simulation = model_simulation(additional_pop[i])
                    additional_obj[i] = multi_obj_func(Obs, simulation, index_metrics)

                population = np.vstack((population, additional_pop))
                objectives = np.vstack((objectives, additional_obj))

            # Early stopping based on improvement criteria
            ii = select_best_solution(objectives)[0]
            current_best_fitness = objectives[ii]
            best_fitness_history.append(current_best_fitness)
            best_individuals.append(population[ii])

            if generation > kstop:
                # Normalize objectives for proper comparison
                normalized_objectives = (objectives - objectives.min(axis=0)) / (objectives.max(axis=0) - objectives.min(axis=0) + 1e-10)
                mean_normalized_fitness = np.mean(np.sum(normalized_objectives, axis=1))
                previous_mean_fitness = np.mean(np.sum((best_fitness_history[-kstop]), axis=0)) if len(best_fitness_history) >= kstop else mean_normalized_fitness
                recent_improvement = (previous_mean_fitness - mean_normalized_fitness) / abs(previous_mean_fitness)
                if recent_improvement < pcento:
                    print(f"Converged at generation {generation} based on improvement criteria.")
                    break

            # Early stopping based on parameter space convergence
            epsilon = 1e-10
            gnrng = np.exp(np.mean(np.log((np.max(population, axis=0) - np.min(population, axis=0) + epsilon) / (ub - lb))))
            if gnrng < peps:
                print(f"Converged at generation {generation} based on parameter space convergence.")
                break

            if generation % (num_generations // 10) == 0:
                print(f"Generation {generation} of {num_generations} completed")
                current_best_fitness
                for j in range(nobj):
                    if mask[j]:
                        print(f"{metrics_name_list[j]}: {current_best_fitness[j]:.3f}")
                    else:
                        print(f"{metrics_name_list[j]}: {(1 - current_best_fitness[j]):.3f}")
    
        # Select the best final solution
        if restart == 0:
            total_objectives = np.vstack((objectives, np.array(best_fitness_history)))
            total_individuals = np.vstack((population, np.array(best_individuals)))
            best_index = select_best_solution(total_objectives)[0]
            best_fitness = total_objectives[best_index]
            best_individual = total_individuals[best_index]
        else:
            total_objectives = np.vstack((objectives, np.array(best_fitness_history)))
            total_individuals = np.vstack((population, np.array(best_individuals)))
            total_objectives = np.vstack((total_objectives, np.array([best_fitness])))
            total_individuals = np.vstack((total_individuals, np.array([best_individual])))
            best_index = select_best_solution(total_objectives)[0]
            best_fitness = total_objectives[best_index]
            best_individual = total_individuals[best_index]
    
    print(f'SPEA2 with tournament selection algorithm completed after {n_restarts} restarts.')
    print('Best fitness found:')
    for j in range(nobj):
        if mask[j]:
            print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
        else:
            print(f"{metrics_name_list[j]}: {(1 - best_fitness[j]):.3f}")
    return best_individual, best_fitness, best_fitness_history

@jit(nopython=True)
def environmental_selection(population, objectives, m):
    """
    Perform environmental selection by calculating fitness and density estimates, selecting individuals for the next generation archive.
    
    Parameters:
    - population: The current population of individuals.
    - objectives: Objective values for each individual.
    - m: Number of neighbors to consider for density estimation.
    
    Returns:
    - archive: The selected archive population.
    - archive_fitness: Fitness values of the individuals in the archive.
    - archive_obj: Objective values of the individuals in the archive.
    """
    npop, nobj = objectives.shape

    min_values = np.full(nobj, np.inf)
    max_values = np.full(nobj, -np.inf)
    
    for i in range(npop):
        for j in range(nobj):
            if objectives[i, j] < min_values[j]:
                min_values[j] = objectives[i, j]
            if objectives[i, j] > max_values[j]:
                max_values[j] = objectives[i, j]
    
    # Normalize objectives to avoid magnitude issues
    normalized_objectives = (objectives - min_values) / (max_values - min_values + 1e-10)
    dist = euclidean_distances(normalized_objectives)

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
        density_estimate[i] = 1 / (2 * k) * (1 / sorted_distances[m+1])

    combined_fitness = fitness_values + density_estimate
    median_fitness = np.median(combined_fitness)
    selection_mask = combined_fitness <= median_fitness
    archive = population[selection_mask, :]
    archive_obj = objectives[selection_mask, :]
    archive_fitness = combined_fitness[selection_mask]

    return archive, archive_fitness, archive_obj

@jit
def crowd_distance(objectives, ranks):
    """
    Calculate crowding distance for each individual in the population.
    
    Parameters:
    - objectives: Objective values for each individual.
    - ranks: Ranks of individuals based on fitness.
    
    Returns:
    - distances: Crowding distance for each individual.
    """
    population_size = objectives.shape[0]
    nobj = objectives.shape[1]
    distances = np.zeros(population_size, dtype=np.float64)

    for rank in range(np.max(ranks) + 1):
        front = np.where(ranks == rank)[0]
        if len(front) == 0:
            continue

        for m in range(nobj):
            sorted_indices = front[np.argsort(objectives[front, m])]
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            min_value = objectives[sorted_indices[0], m]
            max_value = objectives[sorted_indices[-1], m]

            if max_value - min_value == 0:
                continue

            for i in range(1, len(sorted_indices) - 1):
                distances[sorted_indices[i]] += (
                    (objectives[sorted_indices[i + 1], m] - objectives[sorted_indices[i - 1], m])
                    / (max_value - min_value)
                )

    return distances

@jit(nopython=True)
def tournament_selection_with_crowding(ranks, crowding_distances, pressure):
    """
    Perform tournament selection with crowding distance as a tiebreaker.
    
    Parameters:
    - ranks: Ranks of individuals based on fitness.
    - crowding_distances: Crowding distances for each individual.
    - pressure: Selection pressure for tournament selection.
    
    Returns:
    - selected_indices: Indices of the individuals selected for the mating pool.
    """
    n_select = len(ranks)
    n_random = n_select * pressure
    n_perms = math.ceil(n_random / len(ranks))

    P = np.empty((n_random,), dtype=np.int32)
    for i in range(n_perms):
        P[i * len(ranks):(i + 1) * len(ranks)] = np.random.permutation(len(ranks))
    P = P[:n_random].reshape(n_select, pressure)

    selected_indices = np.full(n_select, -1, dtype=np.int32)
    for i in range(n_select):
        a, b = P[i]
        if ranks[a] < ranks[b]:
            selected_indices[i] = a
        elif ranks[a] > ranks[b]:
            selected_indices[i] = b
        else:  # Tie: use crowding distance
            if crowding_distances[a] > crowding_distances[b]:
                selected_indices[i] = a
            else:
                selected_indices[i] = b

    return selected_indices

@jit(nopython=True)
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    """
    Perform crossover operation on the population.
    
    Parameters:
    - population: The current population of individuals.
    - num_vars: Number of variables in each individual.
    - crossover_prob: Probability of crossover.
    - lower_bounds: Lower bounds for each variable.
    - upper_bounds: Upper bounds for each variable.
    
    Returns:
    - child_population: The new population after crossover.
    """
    n_pop = population.shape[0]
    cross_probability = np.random.random(n_pop)
    do_cross = cross_probability < crossover_prob
    R = np.random.randint(0, n_pop, (n_pop, 2))
    parents = R[do_cross]
    child_population = population.copy()

    if num_vars > 1:
        # General case: num_vars > 1
        cross_point = np.random.randint(1, num_vars, len(parents))
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            point = cross_point[i]
            # Concatenate parts of parents to create offspring
            child = np.concatenate((population[parent1, :point], population[parent2, point:]))
            
            # Ensure bounds are respected
            for j in range(num_vars):
                child[j] = min(max(child[j], lower_bounds[j]), upper_bounds[j])
            
            # Update child in the population
            child_population[do_cross][i] = child
    else:
        # Special case: num_vars = 1
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            # For a single variable, offspring is a weighted average of parents
            child = (population[parent1] + population[parent2]) / 2.0
            
            # Ensure bounds are respected
            child = min(max(child[0], lower_bounds[0]), upper_bounds[0])
            
            # Update child in the population
            child_population[do_cross][i] = child

    return child_population

# def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
#     """
#     Perform crossover operation on the population.
    
#     Parameters:
#     - population: The current population of individuals.
#     - num_vars: Number of variables in each individual.
#     - crossover_prob: Probability of crossover.
#     - lower_bounds: Lower bounds for each variable.
#     - upper_bounds: Upper bounds for each variable.
    
#     Returns:
#     - child_population: The new population after crossover.
#     """
#     n_pop = population.shape[0]
#     cross_probability = np.random.random(n_pop)
#     do_cross = cross_probability < crossover_prob
#     R = np.random.randint(0, n_pop, (n_pop, 2))
#     parents = R[do_cross]
#     print('pass 1')
#     print(len(parents))
#     print(num_vars)
#     cross_point = np.random.randint(1, num_vars, len(parents))
#     print('pass 2')
#     child_population = population.copy()

#     for i in range(len(parents)):
#         parent1, parent2 = parents[i]
#         point = cross_point[i]
#         # Concatenate parts of parents to create offspring
#         child = np.concatenate((population[parent1, :point], population[parent2, point:]))

#         # Update the population with the newly generated child
#         for j in range(num_vars):
#             child_population[do_cross][i, j] = min(max(child[j], lower_bounds[j]), upper_bounds[j])

#     return child_population

@jit(nopython=True)
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=20):
    """
    Perform polynomial mutation on the population.
    
    Parameters:
    - population: The current population of individuals.
    - mutation_rate: Probability of mutation.
    - num_vars: Number of variables in each individual.
    - lower_bounds: Lower bounds for each variable.
    - upper_bounds: Upper bounds for each variable.
    - eta_mut: Mutation parameter (default is 20).
    
    Returns:
    - Y: The new population after mutation.
    """
    X = population.copy()
    Y = np.full(X.shape, np.inf)
    do_mutation = np.random.random(X.shape) < mutation_rate
    Y[:, :] = X

    for i in range(len(population)):
        for j in range(num_vars):
            if do_mutation[i, j]:
                xl = lower_bounds[j]
                xu = upper_bounds[j]
                x = X[i, j]

                delta1 = (x - xl) / (xu - xl)
                delta2 = (xu - x) / (xu - xl)
                mut_pow = 1.0 / (eta_mut + 1.0)
                rand = np.random.random()

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_mut + 1.0))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_mut + 1.0))
                    deltaq = 1.0 - (val ** mut_pow)

                mutated_value = x + deltaq * (xu - xl)
                mutated_value = max(xl, min(mutated_value, xu))
                Y[i, j] = mutated_value

    return Y

@jit(nopython=True)
def euclidean_distances(X):
    """
    Calculate the Euclidean distances between individuals in a population.
    
    Parameters:
    - X: The population for which distances are calculated.
    
    Returns:
    - dist: A matrix of pairwise Euclidean distances between individuals.
    """
    # Using broadcasting to calculate the distance matrix efficiently
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e+12)  # Avoid division by zero for self-distances
    return dist