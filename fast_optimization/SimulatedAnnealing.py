import numpy as np
import copy
from .objectives_functions import multi_obj_func

def simulated_annealing(model_simulation, Obs, initialize_population, max_iterations, initial_temperature, cooling_rate, index_metrics):
    """
    Simulated Annealing Optimization Algorithm.
    This algorithm aims to find an optimal solution by probabilistically accepting worse solutions to escape local optima.

    Parameters:
    - model_simulation: Function that simulates the model given an input vector of parameters.
    - Obs: Observed data used for calculating the fitness.
    - initialize_population: Function to initialize the population.
    - max_iterations: Maximum number of iterations to perform.
    - initial_temperature: Starting temperature for annealing.
    - cooling_rate: Rate at which the temperature decreases.
    - index_metrics: Index used for multi-objective evaluation.

    Returns:
    - best_solution: The best solution found.
    - best_fitness: Fitness value of the best solution.
    - fitness_history: History of the best fitness values over iterations.
    """
    # Initialize population with one individual (SA works with a single solution at a time)
    population, lower_bounds, upper_bounds = initialize_population(1)
    current_solution = population[0]
    num_params = current_solution.shape[0]

    # Evaluate initial solution
    current_simulation = model_simulation(current_solution)
    current_fitness = np.array(multi_obj_func(Obs, current_simulation, index_metrics))
    best_solution = copy.deepcopy(current_solution)
    best_fitness = current_fitness.copy()

    fitness_history = [best_fitness.tolist()]
    temperature = initial_temperature

    print('Starting Simulated Annealing optimization...')

    # Main loop
    for iteration in range(max_iterations):
        # Generate a new candidate solution by small random changes
        candidate_solution = current_solution + np.random.uniform(-0.1, 0.1, num_params) * (upper_bounds - lower_bounds)
        candidate_solution = np.clip(candidate_solution, lower_bounds, upper_bounds)

        # Evaluate candidate solution
        candidate_simulation = model_simulation(candidate_solution)
        candidate_fitness = np.array(multi_obj_func(Obs, candidate_simulation, index_metrics))

        # Calculate acceptance probability
        delta_fitness = candidate_fitness - current_fitness
        acceptance_probability = np.exp(-np.sum(delta_fitness) / temperature) if np.any(delta_fitness > 0) else 1.0

        # Decide whether to accept the candidate solution
        if np.random.rand() < acceptance_probability:
            current_solution = candidate_solution
            current_fitness = candidate_fitness

        # Update best solution if the current one is better
        if np.sum(current_fitness) < np.sum(best_fitness):
            best_solution = copy.deepcopy(current_solution)
            best_fitness = current_fitness

        fitness_history.append(best_fitness.tolist())

        # Decrease the temperature
        temperature *= cooling_rate

        # Print progress
        if iteration % (max_iterations // 10) == 0:
            print(f"Iteration {iteration}/{max_iterations}, Best Fitness: {best_fitness}, Temperature: {temperature}")

    return best_solution, best_fitness.tolist(), fitness_history

# Example usage (assuming model_simulation, Obs, initialize_population, and index_metrics are defined):
# best_solution, best_fitness, fitness_history = simulated_annealing(model_simulation, Obs, initialize_population, 1000, 100, 0.95, index_metrics)