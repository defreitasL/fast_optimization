import numpy as np
from numba import njit
import math
from .objectives_functions import multi_obj_func, select_best_solution_L2
from .metrics import backtot

def nsgaii_algorithm_ts(
    model_simulation,
    Obs,
    initialize_population,
    num_generations,
    population_size,
    cross_prob,
    mutation_rate,
    pressure,
    regeneration_rate,
    kstop,
    pcento,
    peps,
    index_metrics,
    n_restarts=5,
):
    """
    NSGA-II with tournament selection (μ+λ environmental selection).

    This routine minimizes multiple objectives produced by `multi_obj_func` when
    the underlying model is evaluated via `model_simulation`. The algorithm uses:
      • Tournament selection with crowding-distance tiebreak,
      • One-point crossover,
      • Polynomial mutation,
      • μ+λ environmental selection (parents ∪ offspring ∪ regenerated),
      • A lightweight evaluation cache to avoid re-simulating duplicates,
      • Optional diversity regeneration each generation.

    Notes
    -----
    • No objective scalarization is performed during evolution; non-dominated
      sorting + crowding controls selection pressure and diversity.
    • The "best" single solution returned is selected via `select_best_solution_L2`
      (distance-to-ideal in normalized objective space), which is robust to
      differently scaled objectives.
    • For speed, identical individuals are evaluated once per restart using a
      simple in-memory cache keyed by rounded parameter vectors.

    Parameters
    ----------
    model_simulation : Callable[[ndarray], Any]
        Function that runs the forward model for a single parameter vector.
        It should return whatever `multi_obj_func` requires to compute objectives.
    Obs : Any
        Observed data passed through to `multi_obj_func` to compute objective values.
    initialize_population : Callable[[int], Tuple[ndarray, ndarray, ndarray]]
        Function that returns a tuple `(population, lower_bounds, upper_bounds)` where:
          - population: (N, D) initial individuals,
          - lower_bounds, upper_bounds: (D,) arrays of parameter bounds.
    num_generations : int
        Number of evolutionary generations per restart.
    population_size : int
        Population size (μ). Offspring are generated to match this size.
    cross_prob : float
        Crossover probability for each individual in the mating pool.
    mutation_rate : float
        Per-variable mutation probability used by polynomial mutation.
    pressure : int
        Tournament size (k) used in selection; k ≥ 2 is typical.
    regeneration_rate : float
        Fraction of the population to inject as newly initialized individuals
        each generation (for diversity).
    kstop : int
        Window length for improvement-based early stopping (not used directly
        here; retained for compatibility).
    pcento : float
        Threshold used in prior early-stopping logic (retained for compatibility).
    peps : float
        Convergence threshold on the normalized geometric parameter range
        (smaller implies tighter convergence).
    index_metrics : Sequence[int]
        Indices selecting which objectives from `multi_obj_func` are active.
    n_restarts : int, optional
        Number of independent runs; the globally best solution is selected among all.

    Returns
    -------
    best_individual : ndarray, shape (D,)
        The single selected compromise solution across all restarts.
    best_fitness : ndarray, shape (M,)
        Objective vector of `best_individual` (M = len(index_metrics)).
    best_fitness_history : List[ndarray]
        Per-generation best objective vectors for the final restart (as tracked).

    See Also
    --------
    fast_non_dominated_sort : Computes Pareto ranks (0 = best) for a set of points.
    crowd_distance : Computes NSGA-II crowding distances per Pareto front.
    tournament_selection_with_crowding : k-way tournament with crowding tiebreak.
    crossover : One-point crossover with bounds enforcement.
    polynomial_mutation : Polynomial mutation with bounds enforcement.

    Notes on Shapes
    ---------------
    • population: (N, D)
    • objectives: (N, M), where M = len(index_metrics)
    • lower_bounds / upper_bounds: (D,)
    """
    print("Precompilation done!")
    print(
        f"Starting NSGA-II with tournament selection algorithm with {n_restarts} restarts..."
    )

    metrics_name_list, mask = backtot()
    metrics_name_list = [metrics_name_list[k] for k in index_metrics]
    mask = [mask[k] for k in index_metrics]

    # all_individuals = np.zeros((0, len(initialize_population(1)[0][0])))
    # all_objectives = np.zeros((0, len(index_metrics)))

    # ---------------------------------------------------------------------
    # Evaluation cache (per restart): avoid re-running identical individuals.
    # The key is a rounded parameter tuple for numerical stability.
    # ---------------------------------------------------------------------
    eval_cache = {}

    def _hash_params(vec):
        return tuple(np.round(np.asarray(vec, dtype=float), 12))

    def _eval_individual(x):
        key = _hash_params(x)
        if key in eval_cache:
            return eval_cache[key]
        sim = model_simulation(x)
        obj = multi_obj_func(Obs, sim, index_metrics)
        eval_cache[key] = obj
        return obj

    for restart in range(n_restarts):
        print(f"Starting {restart+1}/{n_restarts}")

        best_fitness_history = []
        best_individuals = []

        # --------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------
        population, lower_bounds, upper_bounds = initialize_population(population_size)
        n_par = population.shape[1]            # number of decision variables (D)
        n_obj = len(index_metrics)             # number of objectives (M)
        objectives = np.zeros((population_size, n_obj))

        # Evaluate initial population (cached)
        for i in range(population_size):
            objectives[i] = _eval_individual(population[i])

        # Base number of new individuals to inject per generation
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

        # --------------------------------------------------------------
        # Main evolutionary loop
        # --------------------------------------------------------------
        for generation in range(num_generations):
            # Current ranking & crowding (used for logs/diversity gauge)
            ranks, front_indices, front_sizes = fast_non_dominated_sort(objectives)
            crowding_distances = crowd_distance(objectives, ranks)

            # ------------------------------
            # Parent selection (k-way tournament with crowding tiebreak)
            # ------------------------------
            next_population_indices = tournament_selection_with_crowding(
                ranks, crowding_distances, pressure
            ).astype(np.int32)
            mating_pool = population[next_population_indices]

            # ------------------------------
            # Variation: crossover + mutation
            # ------------------------------
            # Adaptive crossover probability (kept ≥ min_cross_prob)
            min_cross_prob = 0.1
            adaptive_cross_prob = max(
                cross_prob * (1 - generation / num_generations), min_cross_prob
            )
            offspring = crossover(
                mating_pool, n_par, adaptive_cross_prob, lower_bounds, upper_bounds
            )

            # Ensure offspring count matches μ
            if offspring.shape[0] != population_size:
                if offspring.shape[0] > population_size:
                    offspring = offspring[:population_size]
                else:
                    # Fill by duplicating random parents (keeps shapes consistent)
                    extra = mating_pool[
                        np.random.randint(
                            0, mating_pool.shape[0], size=population_size - offspring.shape[0]
                        )
                    ]
                    offspring = np.vstack((offspring, extra))

            # Polynomial mutation with adaptive rate (kept ≥ min_mutation_rate)
            min_mutation_rate = 0.01
            adaptive_mutation_rate = max(
                mutation_rate * (1 - generation / num_generations), min_mutation_rate
            )
            offspring = polynomial_mutation(
                offspring, adaptive_mutation_rate, n_par, lower_bounds, upper_bounds
            )

            # ------------------------------
            # Diversity injection (regeneration) and candidate pool
            # ------------------------------
            # Reduce regeneration when crowding already high (cheap diversity heuristic)
            mean_crowd = np.mean(crowding_distances) if np.isfinite(crowding_distances).any() else 0.0
            regen_factor = 0.5 if mean_crowd > 0.5 else 1.0
            num_to_regenerate = int(np.ceil(regeneration_rate * regen_factor * population_size))

            if num_to_regenerate > 0:
                new_individuals, _, _ = initialize_population(num_to_regenerate)
                pool = np.vstack((population, offspring, new_individuals))
            else:
                pool = np.vstack((population, offspring))

            # ------------------------------
            # Evaluate candidate pool with "unique-first" strategy + cache
            # ------------------------------
            unique_pool, unique_idx, inv = np.unique(
                pool, axis=0, return_index=True, return_inverse=True
            )
            unique_obj = np.zeros((unique_pool.shape[0], n_obj))
            for ui in range(unique_pool.shape[0]):
                unique_obj[ui] = _eval_individual(unique_pool[ui])
            pool_objectives = unique_obj[inv]

            # ------------------------------
            # Environmental selection (μ+λ): sort by fronts, truncate by crowding
            # ------------------------------
            ranks_pool, front_indices_pool, front_sizes_pool = fast_non_dominated_sort(pool_objectives)
            crowd_pool = crowd_distance(pool_objectives, ranks_pool)

            new_pop_idx = []
            for rk in np.unique(ranks_pool):
                front_idx = np.where(ranks_pool == rk)[0]
                # If the whole front fits, take it
                if len(new_pop_idx) + len(front_idx) <= population_size:
                    new_pop_idx.extend(front_idx.tolist())
                else:
                    # Truncate this front by descending crowding (keep most diverse)
                    order = np.argsort(-crowd_pool[front_idx])
                    remaining = population_size - len(new_pop_idx)
                    new_pop_idx.extend(front_idx[order[:remaining]].tolist())
                    break

            # Update population/objectives
            population = pool[np.array(new_pop_idx)]
            objectives = pool_objectives[np.array(new_pop_idx)]

            # ------------------------------
            # Tracking: best compromise (distance to ideal in normalized space)
            # ------------------------------
            best_idx = select_best_solution_L2(objectives)[0]
            current_best_fitness = objectives[best_idx]
            best_fitness_history.append(current_best_fitness)
            best_individuals.append(population[best_idx])

            # ------------------------------
            # Convergence: geometric parameter-range contraction
            # ------------------------------
            epsilon = 1e-10
            gnrng = np.exp(
                np.mean(
                    np.log(
                        (np.max(population, axis=0) - np.min(population, axis=0) + epsilon)
                        / (upper_bounds - lower_bounds + epsilon)
                    )
                )
            )
            if gnrng < peps:
                print(
                    f"Converged at generation {generation} based on parameter space convergence."
                )
                break

            # ------------------------------
            # Progress logging (every ~10% of the run)
            # ------------------------------
            if generation % max(1, (num_generations // 10)) == 0:
                print(f"Generation {generation} of {num_generations} completed")
                for j in range(n_obj):
                    if mask[j]:
                        print(f"{metrics_name_list[j]}: {current_best_fitness[j]:.3f}")
                    else:
                        print(f"{metrics_name_list[j]}: {(1 - current_best_fitness[j]):.3f}")

        # --------------------------------------------------------------
        # Per-restart summary and archival for global selection
        # --------------------------------------------------------------
        # pick the per-restart best by L2 to ideal from the tracked history
        hist = np.asarray(best_fitness_history, dtype=float)
        indv = np.asarray(best_individuals, dtype=float)

        # guard in case something odd happens
        if hist.ndim != 2 or hist.size == 0:
            # fallback: take the best in the final population
            restart_best_idx = select_best_solution_L2(objectives)[0]
            restart_best_fit = objectives[restart_best_idx]
            restart_best_ind = population[restart_best_idx]
        else:
            restart_best_idx = select_best_solution_L2(hist)[0]
            restart_best_fit = hist[restart_best_idx]
            restart_best_ind = indv[restart_best_idx]
        # Keep small arrays with *only* restart bests
        if restart == 0:
            restart_bests_obj = restart_best_fit[None, ...]   # (1, M)
            restart_bests_ind = restart_best_ind[None, ...]   # (1, D)
        else:
            restart_bests_obj = np.vstack((restart_bests_obj, restart_best_fit[None, ...]))
            restart_bests_ind = np.vstack((restart_bests_ind, restart_best_ind[None, ...]))
    #     total_objectives = np.vstack((objectives, np.array(best_fitness_history)))
    #     total_individuals = np.vstack((population, np.array(best_individuals)))

    #     if restart > 0:
    #         total_objectives = np.vstack((total_objectives, np.array([best_fitness])))
    #         total_individuals = np.vstack((total_individuals, np.array([best_individual])))

    #     best_index = select_best_solution_L2(total_objectives)[0]
    #     best_fitness = total_objectives[best_index]
    #     best_individual = total_individuals[best_index]

    #     all_individuals = np.vstack((all_individuals, total_individuals))
    #     all_objectives = np.vstack((all_objectives, total_objectives))

    # # --------------------------------------------------------------
    # # Global best across all restarts (robust compromise selection)
    # # --------------------------------------------------------------
    best_index_global = select_best_solution_L2(restart_bests_obj)[0]
    best_individual   = restart_bests_ind[best_index_global]
    best_fitness      = restart_bests_obj[best_index_global]
    # best_index_global = select_best_solution_L2(all_objectives)[0]
    # best_individual = all_individuals[best_index_global]
    # best_fitness = all_objectives[best_index_global]

    print(
        f"NSGA-II with tournament selection algorithm completed after {n_restarts} restarts."
    )
    print("Best fitness found:")
    for j in range(len(index_metrics)):
        if mask[j]:
            print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
        else:
            print(f"{metrics_name_list[j]}: {(1 - best_fitness[j]):.3f}")

    return best_individual, best_fitness, best_fitness_history


@njit
def fast_non_dominated_sort(objectives):
    """
    Fast non-dominated sorting (Deb et al., 2002).

    Parameters
    ----------
    objectives : ndarray, shape (N, M)
        Objective matrix (minimization). N = individuals, M = objectives.

    Returns
    -------
    ranks : ndarray, shape (N,)
        Pareto rank per individual (0 = best front).
    front_indices : ndarray, shape (N, N)
        For each front i (row), indices of its members; unused entries are -1.
    front_sizes : ndarray, shape (N,)
        Size (#members) of each front i.

    Notes
    -----
    An individual p dominates q if p is no worse in all objectives and strictly
    better in at least one objective.
    """
    population_size = objectives.shape[0]
    domination_count = np.zeros(population_size, dtype=np.int32)
    dominated_solutions = np.full((population_size, population_size), -1, dtype=np.int32)
    current_counts = np.zeros(population_size, dtype=np.int32)
    ranks = np.zeros(population_size, dtype=np.int32)

    # Storage for fronts
    front_indices = np.full((population_size, population_size), -1, dtype=np.int32)
    front_sizes = np.zeros(population_size, dtype=np.int32)

    # Identify domination relationships and initial front (rank 0)
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

    # Construct subsequent fronts
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
                    front_indices[i + 1, next_front_size] = q
                    next_front_size += 1
        front_sizes[i + 1] = next_front_size
        i += 1

    return ranks, front_indices, front_sizes


@njit
def crowd_distance(objectives, ranks):
    """
    NSGA-II crowding distance per front.

    Parameters
    ----------
    objectives : ndarray, shape (N, M)
        Objective values (minimization).
    ranks : ndarray, shape (N,)
        Pareto ranks (0 = best).

    Returns
    -------
    distances : ndarray, shape (N,)
        Crowding distance per individual; boundary points get +∞.

    Notes
    -----
    Distances are computed dimension-wise within each front and summed. They are
    normalized by per-objective range to be scale-invariant.
    """
    population_size = objectives.shape[0]
    n_obj = objectives.shape[1]
    distances = np.zeros(population_size, dtype=np.float64)

    for rank in range(np.max(ranks) + 1):
        front = np.where(ranks == rank)[0]
        if len(front) == 0:
            continue

        for m in range(n_obj):
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


@njit
def tournament_selection_with_crowding(ranks, crowding_distances, pressure):
    """
    k-way tournament selection with crowding-distance tiebreak.

    Parameters
    ----------
    ranks : ndarray, shape (N,)
        Pareto ranks (0 = best).
    crowding_distances : ndarray, shape (N,)
        Crowding distance per individual (higher = more diverse).
    pressure : int
        Tournament size (k >= 2).

    Returns
    -------
    selected_indices : ndarray, shape (N,)
        Indices of selected individuals for the mating pool (same count as N).

    Selection Rule
    --------------
    The best candidate is the one with the lowest rank; ties are broken by the
    highest crowding distance.
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
        else:  # Tie: prefer more crowded (greater distance)
            if crowding_distances[a] > crowding_distances[b]:
                selected_indices[i] = a
            else:
                selected_indices[i] = b

    return selected_indices


@njit
def crossover(population, num_vars, crossover_prob, lower_bounds, upper_bounds):
    """
    One-point crossover with bounds enforcement.

    Parameters
    ----------
    population : ndarray, shape (N, D)
        Parent pool from which pairs are sampled.
    num_vars : int
        Number of decision variables (D).
    crossover_prob : float
        Per-individual probability of performing crossover.
    lower_bounds, upper_bounds : ndarray, shape (D,)
        Per-variable bounds applied to offspring.

    Returns
    -------
    child_population : ndarray, shape (N, D)
        New population after crossover (same shape as input).

    Notes
    -----
    Parents are chosen uniformly at random. For D == 1, a simple average is used.
    """
    n_pop = population.shape[0]
    cross_probability = np.random.random(n_pop)
    do_cross = cross_probability < crossover_prob
    R = np.random.randint(0, n_pop, (n_pop, 2))
    parents = R[do_cross]
    child_population = population.copy()

    if num_vars > 1:
        # General case: D > 1
        cross_point = np.random.randint(1, num_vars, len(parents))
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            point = cross_point[i]
            # Build child by concatenating parent segments
            child = np.concatenate((population[parent1, :point], population[parent2, point:]))

            # Clamp to bounds
            for j in range(num_vars):
                child[j] = min(max(child[j], lower_bounds[j]), upper_bounds[j])

            # IMPORTANT: avoid boolean view writes in NumPy (we keep original style)
            child_population[do_cross][i] = child
    else:
        # Special case: D == 1
        for i in range(len(parents)):
            parent1, parent2 = parents[i]
            child = (population[parent1] + population[parent2]) / 2.0
            child = min(max(child[0], lower_bounds[0]), upper_bounds[0])
            child_population[do_cross][i] = child

    return child_population


@njit
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=20):
    """
    Polynomial mutation with bounds enforcement.

    Parameters
    ----------
    population : ndarray, shape (N, D)
        Input population to be mutated.
    mutation_rate : float
        Per-variable mutation probability.
    num_vars : int
        Number of decision variables (D).
    lower_bounds, upper_bounds : ndarray, shape (D,)
        Bounds applied to each mutated variable.
    eta_mut : float, optional
        Distribution index (larger = smaller steps). Default is 20.

    Returns
    -------
    Y : ndarray, shape (N, D)
        Mutated population (same shape as input).
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
