import numpy as np
from numba import jit
from .objectives_functions import multi_obj_func
from .metrics import backtot

def sce_ua_algorithm(model_simulation, Obs, initialize_population, num_generations, population_size, cross_prob, mutation_rate, regeneration_rate, eta_mut, num_complexes, kstop, pcento, peps, index_metrics, n_restarts=5):
    """
    SCE-UA Optimization Algorithm (Adapted to Python).
    This algorithm aims to optimize an objective function by evolving a population using complex-based partitioning and evolving each complex.
    
    Parameters:
    - model_simulation: Function that simulates the model given an input vector of parameters.
    - Obs: Observed data used for calculating the fitness.
    - initialize_population: Function to initialize the population.
    - num_generations: Number of generations to evolve.
    - population_size: Size of the population.
    - cross_prob: Probability of crossover.
    - mutation_rate: Probability of mutation.
    - regeneration_rate: Proportion of the population to be regenerated each generation.
    - eta_mut: Mutation parameter for polynomial mutation.
    - num_complexes: Number of complexes for partitioning the population.
    - kstop: Number of past evolution loops to assess convergence.
    - pcento: Percentage improvement allowed in the past kstop loops for convergence.
    - peps: Threshold for the normalized geometric range of parameters to determine convergence.
    - index_metrics: Index used for multi-objective evaluation.

    Returns:
    - best_individual: The best solution found.
    - best_fitness: Fitness value of the best solution.
    - best_fitness_history: History of the best fitness values over generations.
    """

    best_solution = None
    best_fitness = np.inf

    metrics_name_list, mask = backtot()
    metric_name = [metrics_name_list[k] for k in index_metrics][0]
    mask = [mask[k] for k in index_metrics][0]

    print('Precompilation done!')
    print(f'Starting SCE-UA algorithm with {n_restarts} restarts...')


    for restart in range(n_restarts):

        print(f'Starting {restart+1}/{n_restarts}')


        # Initialize the population
        population, lower_bounds, upper_bounds = initialize_population(population_size)
        num_params = population.shape[1]

        # Evaluate initial population
        fitness_values = np.array([multi_obj_func(Obs, model_simulation(ind), index_metrics)[0] for ind in population])

        # Number of individuals to regenerate each generation
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

        # Archive to store the best solutions found
        archive = []




        # Main loop
        best_fitness_history = []
        best_individuals = []
        for generation in range(num_generations):
            # Adaptively adjust alpha and beta based on generation progress
            alpha = 1.3 - (0.9 * (generation / num_generations))  # Decreases over generations for controlled exploration
            beta = 0.5 + (0.3 * (generation / num_generations))  # Increases for more refined contraction

            # Shuffle the population
            indices = np.random.permutation(population_size)
            population = population[indices]
            fitness_values = fitness_values[indices]

            # Divide population into complexes
            complex_size = population_size // num_complexes
            for complex_index in range(num_complexes):
                start = complex_index * complex_size
                end = start + complex_size if complex_index != num_complexes - 1 else population_size

                complex_population = population[start:end]
                complex_fitness = fitness_values[start:end]

                # Sort complex by fitness
                sorted_indices = np.argsort(complex_fitness)
                complex_population = complex_population[sorted_indices]
                complex_fitness = complex_fitness[sorted_indices]

                # Evolve the complex
                for _ in range(complex_size):
                    # Select simplex by sampling the complex according to a linear probability distribution
                    probabilities = np.linspace(1, 0, complex_size)
                    probabilities /= np.sum(probabilities)
                    simplex_indices = np.random.choice(complex_size, size=num_params + 1, replace=False, p=probabilities)
                    simplex = complex_population[simplex_indices]
                    simplex_fitness = complex_fitness[simplex_indices]

                    # Attempt a reflection point
                    centroid = np.mean(simplex[:-1], axis=0)
                    worst_point = simplex[-1]
                    reflected_point = centroid + alpha * (centroid - worst_point)
                    reflected_point = np.clip(reflected_point, lower_bounds, upper_bounds)
                    reflected_fitness = multi_obj_func(Obs, model_simulation(reflected_point), index_metrics)[0]

                    if reflected_fitness < simplex_fitness[-1]:
                        simplex[-1] = reflected_point
                        simplex_fitness[-1] = reflected_fitness
                    else:
                        # Attempt a contraction point
                        contracted_point = centroid + beta * (worst_point - centroid)
                        contracted_point = np.clip(contracted_point, lower_bounds, upper_bounds)
                        contracted_fitness = multi_obj_func(Obs, model_simulation(contracted_point), index_metrics)[0]

                        if contracted_fitness < simplex_fitness[-1]:
                            simplex[-1] = contracted_point
                            simplex_fitness[-1] = contracted_fitness
                        else:
                            # Replace with a random point
                            random_point, _, _ = initialize_population(1)
                            simplex[-1] = random_point[0]
                            simplex_fitness[-1] = multi_obj_func(Obs, model_simulation(random_point[0]), index_metrics)[0]

                    # Update the complex with the evolved simplex
                    complex_population[simplex_indices] = simplex
                    complex_fitness[simplex_indices] = simplex_fitness

                # Update the main population with the evolved complex
                population[start:end] = complex_population
                fitness_values[start:end] = complex_fitness

            # Elitism: Keep the best solution found so far
            current_best_index = np.argmin(fitness_values)
            current_best_fitness = fitness_values[current_best_index]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[current_best_index]
                archive.append((best_solution, best_fitness))

            # Crossover operation
            cross_prob = cross_prob * (1 - generation / num_generations)  # Gradually reduce crossover rate
            population = crossover(population, num_params, cross_prob, lower_bounds, upper_bounds)

            # Mutation operation
            mutation_rate = mutation_rate * (1 - generation / num_generations)  # Gradually reduce mutation rate
            population = polynomial_mutation(population, mutation_rate, num_params, lower_bounds, upper_bounds, eta_mut)

            # Reintroduce new individuals to maintain genetic diversity
            new_individuals, _, _ = initialize_population(num_to_regenerate)
            new_individuals_fitness = np.array([multi_obj_func(Obs, model_simulation(ind), index_metrics)[0] for ind in new_individuals])

            # Replace worst individuals with new ones
            worst_indices = np.argsort(fitness_values)[-num_to_regenerate:]
            population[worst_indices] = new_individuals
            fitness_values[worst_indices] = new_individuals_fitness

            best_fitness_history.append(best_fitness)
            best_individuals.append(best_solution)

            # Simulated annealing: Accept worse solutions with decreasing probability
            temperature = max(0.1, (1 - generation / num_generations))  # Temperature decreases over time
            if np.random.rand() < temperature:
                random_index = np.random.randint(population_size)
                random_solution = population[random_index]
                random_fitness = multi_obj_func(Obs, model_simulation(random_solution), index_metrics)[0]
                if random_fitness < best_fitness:
                    best_fitness = random_fitness
                    best_solution = random_solution

            # Check convergence based on improvement criteria or parameter space convergence
            if generation > kstop:
                recent_improvement = (best_fitness_history[-kstop] - best_fitness) / abs(best_fitness_history[-kstop])
                if recent_improvement < pcento:
                    print(f"Converged at generation {generation} based on improvement criteria.")
                    break

            gnrng = np.exp(np.mean(np.log((np.max(population, axis=0) - np.min(population, axis=0)) / (upper_bounds - lower_bounds))))
            if gnrng < peps:
                print(f"Converged at generation {generation} based on parameter space convergence.")
                break

            if generation % (num_generations // 10) == 0:
                print(f"Generation {generation} of {num_generations} completed.")
                if mask:
                    print(f"{metric_name}: {best_fitness:.3f}")
                else:
                    print(f"{metric_name}: {(1-best_fitness):.3f}")
        
        # Select the best final solution
        total_objectives = np.hstack((fitness_values, np.array(best_fitness_history).flatten()))
        total_individuals = np.vstack((population, np.array(best_individuals)))
        best_index = np.argmin(total_objectives)
        best_fitness = total_objectives[best_index]
        best_individual = total_individuals[best_index]

    print(f'SCE-UA completed after {n_restarts} restarts.')
    print('Best fitness found:')
    if mask:
        print(f"{metric_name}: {best_fitness:.3f}")
    else:
        print(f"{metric_name}: {(1-best_fitness):.3f}")


    # Return the best solution from the archive
    return best_individual, best_fitness, best_fitness_history


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
#     cross_point = np.random.randint(1, num_vars, len(parents))
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
