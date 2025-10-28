import numpy as np
from numba import njit
from .objectives_functions import multi_obj_func
from .metrics import backtot

def sce_ua_algorithm(
    model_simulation,
    Obs,
    initialize_population,
    num_generations,
    population_size,
    cross_prob,
    mutation_rate,
    regeneration_rate,
    eta_mut,
    num_complexes,
    kstop,
    pcento,
    peps,
    index_metrics,
    n_restarts=5,
):
    """
    SCE-UA (Shuffled Complex Evolution) single-objective calibration.

    This implementation follows the classic SCE-UA idea:
      1) Initialize a population.
      2) Shuffle and split into 'complexes'.
      3) Within each complex, repeatedly evolve a small simplex via
         reflection / contraction / random replacement (à la Nelder–Mead),
         biased toward better points for sampling.
      4) Reassemble, apply genetic operators (crossover, mutation),
         and inject random individuals (regeneration) to preserve diversity.
      5) Track the global best and check convergence.

    Notes on objective direction:
      * `multi_obj_func` already returns a value to MINIMIZE (it flips "maximize" metrics
        via 1 - metric). Here we always minimize the first (and only) metric specified
        by `index_metrics`.

    Parameters
    ----------
    model_simulation : callable
        f(theta) -> model outputs used to compute the objective.
    Obs : array-like
        Observations used by the metric.
    initialize_population : callable
        f(n) -> (pop, lb, ub), where pop.shape == (n, npar).
    num_generations : int
        Maximum number of generations (outer loops).
    population_size : int
        Population size.
    cross_prob : float
        Crossover probability (will be adaptively decreased).
    mutation_rate : float
        Mutation probability (will be adaptively decreased).
    regeneration_rate : float
        Fraction of population replaced each generation with random candidates.
    eta_mut : float
        Polynomial mutation parameter.
    num_complexes : int
        Number of complexes to split the population into each generation.
    kstop : int
        Window length for improvement-based early stopping.
    pcento : float
        Minimum relative improvement required over the last `kstop` steps to continue.
    peps : float
        Threshold for normalized geometric range of parameters for convergence.
    index_metrics : list[int]
        Single-element list with the metric index (see your `backtot()`/`opt()`).
    n_restarts : int, optional
        Number of multi-start restarts.

    Returns
    -------
    best_individual : ndarray (npar,)
        Best parameters found across restarts.
    best_fitness : float
        Objective (to minimize) of the best individual.
    best_fitness_history : list[float]
        Best-of-run fitness per generation for the last restart.
    """
    # Resolve metric name and direction (mask=True means minimize in your framework)
    metrics_name_list, mask_list = backtot()
    metric_name = [metrics_name_list[k] for k in index_metrics][0]
    is_minimize = [mask_list[k] for k in index_metrics][0]

    print("Precompilation done!")
    print(f"Starting SCE-UA algorithm with {n_restarts} restarts...")

    # Global best across restarts
    global_best_individual = None
    global_best_fitness = np.inf

    for restart in range(n_restarts):
        print(f"Starting {restart + 1}/{n_restarts}")

        # --- 1) Initialize population and bounds
        population, lower_bounds, upper_bounds = initialize_population(population_size)
        num_params = population.shape[1]

        # Evaluate initial population (single-objective: take [0])
        fitness_values = np.array(
            [multi_obj_func(Obs, model_simulation(ind), index_metrics)[0] for ind in population]
        )

        # Derived sizes
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))
        best_fitness_history = []
        best_individuals = []

        # Keep the best-so-far for this restart
        best_solution = population[np.argmin(fitness_values)].copy()
        best_fitness = np.min(fitness_values)

        # --- 2) Main generations loop
        for generation in range(num_generations):
            # Adaptive reflection/contraction coefficients
            alpha = 1.3 - (0.9 * (generation / num_generations))  # decreases over time
            beta = 0.5 + (0.3 * (generation / num_generations))   # increases over time

            # Shuffle population and fitness in the same way
            perm = np.random.permutation(population_size)
            population = population[perm]
            fitness_values = fitness_values[perm]

            # Split into complexes
            complex_size = max(2, population_size // max(1, num_complexes))
            for complex_index in range(num_complexes):
                start = complex_index * complex_size
                end = start + complex_size if complex_index != num_complexes - 1 else population_size

                complex_population = population[start:end]
                complex_fitness = fitness_values[start:end]

                # Sort complex by fitness ascending (best first)
                order = np.argsort(complex_fitness)
                complex_population = complex_population[order]
                complex_fitness = complex_fitness[order]

                # Evolve the complex
                for _ in range(complex_population.shape[0]):
                    # Sample a simplex (size = min(num_params+1, complex_size)) with bias to good points
                    csize = complex_population.shape[0]
                    simplex_size = min(num_params + 1, csize)
                    # Linear probability: higher for better-ranked individuals
                    probs = np.linspace(1.0, 0.0, csize)
                    s = probs.sum()
                    if s <= 1e-12:
                        probs[:] = 1.0 / csize
                    else:
                        probs /= s
                    simplex_indices = np.random.choice(
                        csize, size=simplex_size, replace=False, p=probs
                    )

                    simplex = complex_population[simplex_indices].copy()
                    simplex_fitness = complex_fitness[simplex_indices].copy()

                    # Identify TRUE worst point in the sampled simplex
                    worst_local_idx = np.argmax(simplex_fitness)
                    worst_point = simplex[worst_local_idx]
                    worst_f = simplex_fitness[worst_local_idx]

                    # Centroid of all but the worst
                    centroid = np.mean(np.delete(simplex, worst_local_idx, axis=0), axis=0)

                    # Reflect
                    reflected_point = centroid + alpha * (centroid - worst_point)
                    reflected_point = np.clip(reflected_point, lower_bounds, upper_bounds)
                    reflected_f = multi_obj_func(Obs, model_simulation(reflected_point), index_metrics)[0]

                    if reflected_f < worst_f:
                        # Accept reflection
                        simplex[worst_local_idx] = reflected_point
                        simplex_fitness[worst_local_idx] = reflected_f
                    else:
                        # Contract toward centroid from worst
                        contracted_point = centroid + beta * (worst_point - centroid)
                        contracted_point = np.clip(contracted_point, lower_bounds, upper_bounds)
                        contracted_f = multi_obj_func(Obs, model_simulation(contracted_point), index_metrics)[0]

                        if contracted_f < worst_f:
                            simplex[worst_local_idx] = contracted_point
                            simplex_fitness[worst_local_idx] = contracted_f
                        else:
                            # Random replacement within bounds
                            random_point, _, _ = initialize_population(1)
                            rp = random_point[0]
                            simplex[worst_local_idx] = rp
                            simplex_fitness[worst_local_idx] = multi_obj_func(
                                Obs, model_simulation(rp), index_metrics
                            )[0]

                    # Write back evolved simplex into the complex
                    complex_population[simplex_indices] = simplex
                    complex_fitness[simplex_indices] = simplex_fitness

                # Update the main arrays for this complex slot
                population[start:end] = complex_population
                fitness_values[start:end] = complex_fitness

            # --- 3) Elitism: keep global best of this restart
            curr_best_idx = np.argmin(fitness_values)
            curr_best_fit = fitness_values[curr_best_idx]
            if curr_best_fit < best_fitness:
                best_fitness = curr_best_fit
                best_solution = population[curr_best_idx].copy()

            # --- 4) Genetic operators (with adaptive rates)
            # (a) Crossover
            cross_prob_gen = cross_prob * (1.0 - generation / num_generations)
            population = crossover(population, num_params, cross_prob_gen, lower_bounds, upper_bounds)

            # (b) Mutation
            mutation_rate_gen = mutation_rate * (1.0 - generation / num_generations)
            population = polynomial_mutation(
                population, mutation_rate_gen, num_params, lower_bounds, upper_bounds, eta_mut
            )

            # Re-evaluate fitness AFTER GA operators
            fitness_values = np.array(
                [multi_obj_func(Obs, model_simulation(ind), index_metrics)[0] for ind in population]
            )

            # --- 5) Regeneration (inject diversity)
            if num_to_regenerate > 0:
                new_individuals, _, _ = initialize_population(num_to_regenerate)
                new_fitness = np.array(
                    [multi_obj_func(Obs, model_simulation(ind), index_metrics)[0] for ind in new_individuals]
                )

                # Replace current worst individuals
                worst_indices = np.argsort(fitness_values)[-num_to_regenerate:]
                population[worst_indices] = new_individuals
                fitness_values[worst_indices] = new_fitness

            # Track best history (for this restart)
            best_fitness_history.append(best_fitness)
            best_individuals.append(best_solution.copy())

            # Optional SA-like occasional acceptance (kept simple and conservative)
            temperature = max(0.1, (1.0 - generation / num_generations))
            if np.random.rand() < temperature:
                ridx = np.random.randint(population_size)
                rf = multi_obj_func(Obs, model_simulation(population[ridx]), index_metrics)[0]
                if rf < best_fitness:
                    best_fitness = rf
                    best_solution = population[ridx].copy()

            # --- 6) Convergence checks
            if generation > kstop:
                prev_best = best_fitness_history[-kstop]
                if np.abs(prev_best) > 0:
                    recent_improvement = (prev_best - best_fitness) / np.abs(prev_best)
                else:
                    recent_improvement = 0.0
                if recent_improvement < pcento:
                    print(f"Converged at generation {generation} based on improvement criteria.")
                    break

            eps = 1e-12
            ranges = np.maximum(np.max(population, axis=0) - np.min(population, axis=0), eps)
            denom = np.maximum(upper_bounds - lower_bounds, eps)
            gnrng = np.exp(np.mean(np.log(ranges / denom)))
            if gnrng < peps:
                print(f"Converged at generation {generation} based on parameter space convergence.")
                break

            if generation % max(1, (num_generations // 10)) == 0:
                print(f"Generation {generation} of {num_generations} completed.")
                if is_minimize:
                    print(f"{metric_name}: {best_fitness:.3f}")
                else:
                    # If the underlying metric is "maximize", multi_obj_func produced (1 - metric)
                    # so printing (1 - best_fitness) recovers the metric scale
                    print(f"{metric_name}: {(1.0 - best_fitness):.3f}")

        # --- 7) Final selection for this restart
        # Combine current population and the tracked best across generations
        combined_fitness = np.concatenate(
            [fitness_values, np.asarray(best_fitness_history, dtype=float)]
        )
        combined_individuals = np.vstack(
            [population, np.asarray(best_individuals)]
        )
        best_idx = np.argmin(combined_fitness)
        restart_best_fitness = combined_fitness[best_idx]
        restart_best_individual = combined_individuals[best_idx].copy()

        # Update global best across restarts
        if restart_best_fitness < global_best_fitness:
            global_best_fitness = restart_best_fitness
            global_best_individual = restart_best_individual.copy()

    # --- 8) Report
    print("SCE-UA completed after {} restarts.".format(n_restarts))
    print("Best fitness found:")
    if is_minimize:
        print(f"{metric_name}: {global_best_fitness:.3f}")
    else:
        print(f"{metric_name}: {(1.0 - global_best_fitness):.3f}")

    return global_best_individual, global_best_fitness, best_fitness_history


@njit
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    """
    One-point crossover with per-individual application.

    Notes
    -----
    - A crossover is attempted independently for each individual with
      probability `crossover_prob`. For each selected individual, two
      random parents are chosen and a one-point crossover is performed.
    - Bounds are enforced (clamped) gene-wise on the resulting child.
    - Uses integer indices instead of boolean views to avoid temporary
      copies under Numba and ensure in-place updates.

    Parameters
    ----------
    population : (N, D) array
        Current population (N individuals, D parameters).
    num_vars : int
        Number of parameters (D).
    crossover_prob : float
        Probability that a given individual will be replaced by a crossover child.
    lower_bounds, upper_bounds : (D,) arrays
        Gene-wise lower/upper bounds.

    Returns
    -------
    child_population : (N, D) array
        Population after applying crossover (in-place copy of `population`).
    """
    n_pop = population.shape[0]
    # Decide who gets crossover
    cross_probability = np.random.random(n_pop)
    do_cross = cross_probability < crossover_prob
    idxs = np.where(do_cross)[0]  # integer indices of those to replace

    # For each selected individual, draw two parents
    parents = np.random.randint(0, n_pop, (idxs.shape[0], 2))

    # Copy base pop for output (children overwrite selected rows)
    child_population = population.copy()

    if num_vars > 1:
        # One-point crossover index in [1, num_vars-1]
        cross_point = np.random.randint(1, num_vars, idxs.shape[0])

        for i in range(idxs.shape[0]):
            p1, p2 = parents[i]
            point = cross_point[i]

            # Build child gene-wise (avoid np.concatenate for clarity under Numba)
            child = np.empty(num_vars, dtype=population.dtype)
            for j in range(point):
                child[j] = population[p1, j]
            for j in range(point, num_vars):
                child[j] = population[p2, j]

            # Clamp to bounds
            for j in range(num_vars):
                if child[j] < lower_bounds[j]:
                    child[j] = lower_bounds[j]
                elif child[j] > upper_bounds[j]:
                    child[j] = upper_bounds[j]

            # Write back into the selected slot
            child_population[idxs[i], :] = child

    else:
        # Special case D == 1: child = average of two parents, clamped
        for i in range(idxs.shape[0]):
            p1, p2 = parents[i]
            val = 0.5 * (population[p1, 0] + population[p2, 0])

            # Clamp
            if val < lower_bounds[0]:
                val = lower_bounds[0]
            elif val > upper_bounds[0]:
                val = upper_bounds[0]

            child_population[idxs[i], 0] = val

    return child_population

@njit
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=20):
    """
    Per-gene polynomial mutation (Deb, 2001).

    Notes
    -----
    - Mutation is applied independently to each gene with probability `mutation_rate`.
    - If a gene's span is zero (upper == lower), the mutation is skipped to avoid
      division-by-zero and NaNs.
    - The mutated gene is clamped to [lower_bounds[j], upper_bounds[j]].

    Parameters
    ----------
    population : (N, D) array
        Current population (N individuals, D parameters).
    mutation_rate : float
        Per-gene mutation probability in [0, 1].
    num_vars : int
        Number of parameters (D).
    lower_bounds, upper_bounds : (D,) arrays
        Gene-wise lower/upper bounds.
    eta_mut : float, optional
        Distribution index for the polynomial mutation (controls tail heaviness).

    Returns
    -------
    Y : (N, D) array
        Population after mutation (copy of `population` with mutated genes).
    """
    X = population.copy()
    Y = X.copy()  # start from original; modify where mutation happens

    # Bernoulli mask: which genes mutate
    do_mutation = np.random.random(X.shape) < mutation_rate

    for i in range(X.shape[0]):
        for j in range(num_vars):
            if do_mutation[i, j]:
                xl = lower_bounds[j]
                xu = upper_bounds[j]

                # Skip if the gene is effectively fixed (no span)
                span = xu - xl
                if span <= 1e-12:
                    continue

                x = X[i, j]
                # Normalize distance to bounds
                delta1 = (x - xl) / span
                delta2 = (xu - x) / span

                # Standard polynomial mutation draw
                mut_pow = 1.0 / (eta_mut + 1.0)
                rand = np.random.random()

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_mut + 1.0))
                    delta_q = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_mut + 1.0))
                    delta_q = 1.0 - (val ** mut_pow)

                mutated_value = x + delta_q * span

                # Clamp to bounds
                if mutated_value < xl:
                    mutated_value = xl
                elif mutated_value > xu:
                    mutated_value = xu

                Y[i, j] = mutated_value

    return Y