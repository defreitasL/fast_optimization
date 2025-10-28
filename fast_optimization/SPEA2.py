import numpy as np
from numba import njit
import math
from .objectives_functions import multi_obj_func, select_best_solution_L2 #,, select_best_from_first_front
from .metrics import backtot

def spea2_algorithm(
    model_simulation,
    Obs,
    initialize_population,
    num_generations,
    population_size,
    cross_prob,
    mutation_rate,
    pressure,
    regeneration_rate,
    m,
    eta_mut,
    kstop,
    pcento,
    peps,
    index_metrics,
    n_restarts=5,
):
    """
    Strength Pareto Evolutionary Algorithm 2 (SPEA2).

    This routine minimizes multiple objectives returned by `multi_obj_func` for
    simulations produced by `model_simulation`. It implements the canonical
    SPEA2 loop:
      • Environmental selection (dominance-based fitness + density),
      • Archive truncation (if needed),
      • Tournament selection with crowding-distance tiebreak on the archive,
      • One-point crossover and polynomial mutation,
      • Optional diversity injection (regeneration),
      • Archive-driven replacement (new population is the selected archive),
      • Optional convergence checks.

    Notes
    -----
    • Objective values are assumed to be in MINIMIZATION sense.
    • Environmental selection follows the SPEA2 definitions:
        S(i)  : strength (count of solutions dominated by i),
        R(i)  : raw fitness (sum of S(j) for solutions dominating i),
        D(i)  : density via distance to k-th nearest neighbor,
        F(i)  : final fitness = R(i) + D(i), lower is better.
    • The final single "best" solution is selected by `select_best_solution_L2`
      (distance to ideal in normalized objective space), which is robust across
      differently scaled objectives.

    Parameters
    ----------
    model_simulation : Callable[[ndarray], Any]
        Forward model function. Receives one parameter vector and returns a
        simulation object consumed by `multi_obj_func`.
    Obs : Any
        Observations passed to `multi_obj_func` to compute objective values.
    initialize_population : Callable[[int], Tuple[ndarray, ndarray, ndarray]]
        Function returning `(population, lower_bounds, upper_bounds)` where:
          - population     : (N, D) initial parameter vectors,
          - lower_bounds   : (D,) per-variable lower bounds,
          - upper_bounds   : (D,) per-variable upper bounds.
    num_generations : int
        Number of generations per restart.
    population_size : int
        Population size (μ). The archive is truncated to this size if larger.
    cross_prob : float
        Per-individual probability of performing crossover.
    mutation_rate : float
        Per-variable mutation probability in polynomial mutation.
    pressure : int
        Tournament size (k ≥ 2) used in selection from the archive.
    regeneration_rate : float
        Fraction of new random individuals injected each generation.
    m : int
        Neighbor index used in density estimation (k-th nearest neighbor).
    eta_mut : float
        Distribution index for polynomial mutation (larger = smaller steps).
    kstop : int
        Length of the improvement window (maintained for compatibility).
    pcento : float
        Threshold for improvement-based stopping (maintained for compatibility).
    peps : float
        Convergence threshold on normalized geometric parameter range.
    index_metrics : Sequence[int]
        Indices selecting which objectives from `multi_obj_func` are active.
    n_restarts : int, optional
        Number of independent restarts; best across restarts is returned.

    Returns
    -------
    best_individual : ndarray, shape (D,)
        Selected compromise solution after all restarts.
    best_fitness : ndarray, shape (M,)
        Objective vector for `best_individual` (M = len(index_metrics)).
    best_fitness_history : List[ndarray]
        Per-generation best objective vectors for the last restart.

    See Also
    --------
    environmental_selection : SPEA2 fitness + density archive selection.
    crowd_distance : NSGA-II crowding (used for selection tiebreak).
    tournament_selection_with_crowding : k-way tournament on archive.
    crossover : One-point crossover with bounds enforcement.
    polynomial_mutation : Polynomial mutation with bounds enforcement.

    Shape Conventions
    -----------------
    • population: (N, D)
    • objectives: (N, M) with M = len(index_metrics)
    • bounds:    (D,)
    """
    print("Precompilation done!")
    print(f"Starting SPEA2 algorithm with {n_restarts} restarts...")

    metrics_name_list, mask = backtot()
    metrics_name_list = [metrics_name_list[k] for k in index_metrics]
    mask = [mask[k] for k in index_metrics]

    for restart in range(n_restarts):
        print(f"Starting {restart+1}/{n_restarts}")

        best_fitness_history = []
        best_individuals = []

        # --------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------
        population, lb, ub = initialize_population(population_size)
        npar = population.shape[1]            # number of decision variables (D)
        nobj = len(index_metrics)             # number of objectives (M)
        objectives = np.zeros((population_size, nobj))

        # Evaluate initial population
        for i in range(population_size):
            simulation = model_simulation(population[i])
            objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

        # Number of new individuals to inject per generation
        num_to_regenerate = int(np.ceil(regeneration_rate * population_size))

        # --------------------------------------------------------------
        # Main SPEA2 loop
        # --------------------------------------------------------------
        for generation in range(num_generations):
            # 1) Environmental selection → archive (dominance + density)
            archive, archive_fitness, archive_objectives = environmental_selection(
                population, objectives, m
            )

            # Guard against empty archive (should be rare)
            if len(archive) == 0:
                sorted_indices = np.argsort(archive_fitness)
                archive = population[sorted_indices[:1], :]
                archive_objectives = objectives[sorted_indices[:1], :]
                archive_fitness = archive_fitness[sorted_indices[:1]]

            # 2) Truncate archive to population size if needed
            if len(archive) > population_size:
                archive_indices = np.argsort(archive_fitness)[:population_size]
                archive = archive[archive_indices]
                archive_objectives = archive_objectives[archive_indices]
                archive_fitness = archive_fitness[archive_indices]

            # 3) Parent selection on the ARCHIVE (k-way tournament + crowding)
            order = np.argsort(archive_fitness)      # best first
            ranks = np.empty_like(order)              # convert to rank labels
            for i in range(order.size):
                ranks[order[i]] = i
            crowding_distances = crowd_distance(archive_objectives, ranks)
            pool_indexes = tournament_selection_with_crowding(
                ranks, crowding_distances, pressure
            )
            mating_pool = archive[pool_indexes]

            # 4) Variation: crossover + polynomial mutation
            min_cross_prob = 0.1  # keep some recombination pressure
            adaptive_cross_prob = max(
                cross_prob * (1 - generation / num_generations), min_cross_prob
            )
            offspring = crossover(mating_pool, npar, adaptive_cross_prob, lb, ub)

            min_mutation_rate = 0.01  # avoid premature convergence
            adaptive_mutation_rate = max(
                mutation_rate * (1 - generation / num_generations), min_mutation_rate
            )
            offspring = polynomial_mutation(offspring, adaptive_mutation_rate, npar, lb, ub, eta_mut)

            # 5) Evaluate offspring
            offspring_objectives = np.zeros((len(offspring), nobj))
            for i in range(len(offspring)):
                simulation = model_simulation(offspring[i])
                offspring_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            # 6) Regeneration (diversity injection)
            new_individuals, _, _ = initialize_population(num_to_regenerate)
            new_individuals_objectives = np.zeros((num_to_regenerate, nobj))
            for i in range(num_to_regenerate):
                simulation = model_simulation(new_individuals[i])
                new_individuals_objectives[i] = multi_obj_func(Obs, simulation, index_metrics)

            # 7) Candidate union (archive + offspring + regenerated)
            candidates = np.vstack((archive, offspring, new_individuals))
            candidates_objectives = np.vstack(
                (archive_objectives, offspring_objectives, new_individuals_objectives)
            )

            # 8) Environmental selection over the union (canonical SPEA2)
            archive, archive_fitness, archive_objectives = environmental_selection(
                candidates, candidates_objectives, m
            )

            # 9) Truncate archive if larger than μ
            if len(archive) > population_size:
                archive_indices = np.argsort(archive_fitness)[:population_size]
                archive = archive[archive_indices]
                archive_objectives = archive_objectives[archive_indices]
                archive_fitness = archive_fitness[archive_indices]

            # 10) Replacement: next population is the archive
            population = archive
            objectives = archive_objectives

            # Keep population size consistent (rarely needed)
            if len(population) < population_size:
                additional_pop, _, _ = initialize_population(population_size - len(population))
                additional_obj = np.zeros((population_size - len(population), nobj))
                for i in range(len(additional_pop)):
                    simulation = model_simulation(additional_pop[i])
                    additional_obj[i] = multi_obj_func(Obs, simulation, index_metrics)

                population = np.vstack((population, additional_pop))
                objectives = np.vstack((objectives, additional_obj))

            # Track best archive member (for history/diagnostics)
            ii = select_best_solution_L2(objectives)[0]
            current_best_fitness = objectives[ii]
            best_fitness_history.append(current_best_fitness)
            best_individuals.append(population[ii])

            # Optional improvement-based stopping (kept for compatibility)
            if generation > kstop:
                normalized_objectives = (objectives - objectives.min(axis=0)) / (
                    objectives.max(axis=0) - objectives.min(axis=0) + 1e-10
                )
                mean_normalized_fitness = np.mean(np.sum(normalized_objectives, axis=1))
                previous_mean_fitness = (
                    np.mean(np.sum((best_fitness_history[-kstop]), axis=0))
                    if len(best_fitness_history) >= kstop
                    else mean_normalized_fitness
                )
                recent_improvement = (previous_mean_fitness - mean_normalized_fitness) / abs(
                    previous_mean_fitness
                )
                if recent_improvement < pcento:
                    print(f"Converged at generation {generation} based on improvement criteria.")
                    break

            # Parameter-space contraction stopping (robust, cheap)
            epsilon = 1e-10
            gnrng = np.exp(
                np.mean(
                    np.log(
                        (np.max(population, axis=0) - np.min(population, axis=0) + epsilon)
                        / (ub - lb + epsilon)
                    )
                )
            )
            if gnrng < peps:
                print(f"Converged at generation {generation} based on parameter space convergence.")
                break

            # Progress log (every ~10% of run)
            if generation % (num_generations // 10) == 0:
                print(f"Generation {generation} of {num_generations} completed")
                for j in range(nobj):
                    if mask[j]:
                        print(f"{metrics_name_list[j]}: {current_best_fitness[j]:.3f}")
                    else:
                        print(f"{metrics_name_list[j]}: {(1 - current_best_fitness[j]):.3f}")

        # --------------------------------------------------------------
        # Per-restart summary and archival for global best selection
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


    print("Best fitness found:")
    for j in range(nobj):
        if mask[j]:
            print(f"{metrics_name_list[j]}: {best_fitness[j]:.3f}")
        else:
            print(f"{metrics_name_list[j]}: {(1 - best_fitness[j]):.3f}")

    return best_individual, best_fitness, best_fitness_history


@njit
def environmental_selection(population, objectives, m):
    """
    SPEA2 environmental selection (canonical).

    Computes archive membership using dominance-based fitness plus density in
    normalized objective space. The final fitness is:
        F(i) = R(i) + D(i),
    where R is raw fitness (sum of strength of dominators) and D is a density
    term based on the distance to the k-th nearest neighbor (k = m).

    Parameters
    ----------
    population : ndarray, shape (N, D)
        Current population.
    objectives : ndarray, shape (N, M)
        Objective matrix (minimization).
    m : int
        k for the k-th nearest neighbor used in density estimation.

    Returns
    -------
    archive : ndarray, shape (A, D)
        Selected archive population (F <= median(F)), A ≥ 1.
    archive_fitness : ndarray, shape (A,)
        Final fitness values F(i) for the archive.
    archive_obj : ndarray, shape (A, M)
        Objective vectors of the archive members.

    Notes
    -----
    • Objectives are normalized per dimension by min–max scaling.
    • If the archive would be empty, at least the best individual by F is kept.
    """
    npop, nobj = objectives.shape

    # ---- Min–max normalization per objective (scale invariance) ----
    min_vals = np.empty(nobj)
    max_vals = np.empty(nobj)
    for j in range(nobj):
        mn = objectives[0, j]
        mx = objectives[0, j]
        for i in range(1, npop):
            v = objectives[i, j]
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        min_vals[j] = mn
        max_vals[j] = mx

    norm_obj = np.empty_like(objectives)
    for j in range(nobj):
        span = max_vals[j] - min_vals[j]
        if span <= 1e-12:
            for i in range(npop):
                norm_obj[i, j] = 0.0
        else:
            inv = 1.0 / span
            for i in range(npop):
                norm_obj[i, j] = (objectives[i, j] - min_vals[j]) * inv

    # ---- Pairwise dominance (minimization) ----
    dominate = np.zeros((npop, npop), dtype=np.uint8)
    for i in range(npop):
        for j in range(npop):
            if i == j:
                continue
            not_worse = True
            strictly_better = False
            for k in range(nobj):
                if norm_obj[i, k] > norm_obj[j, k]:
                    not_worse = False
                    break
                elif norm_obj[i, k] < norm_obj[j, k]:
                    strictly_better = True
            if not_worse and strictly_better:
                dominate[i, j] = 1

    # Strength S(i): number dominated by i
    S = np.zeros(npop)
    for i in range(npop):
        cnt = 0.0
        for j in range(npop):
            if dominate[i, j] == 1:
                cnt += 1.0
        S[i] = cnt

    # Raw fitness R(i): sum of strengths of dominators of i
    R = np.zeros(npop)
    for i in range(npop):
        s = 0.0
        for j in range(npop):
            if dominate[j, i] == 1:
                s += S[j]
        R[i] = s

    # ---- Density D(i): inverse of (sigma_k + 2) with k-th nearest neighbor ----
    diff = norm_obj[:, np.newaxis, :] - norm_obj[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    for i in range(npop):
        dist[i, i] = 1e12  # exclude self

    k_nn = m
    max_k = npop - 2
    if k_nn > max_k:
        k_nn = max_k if max_k >= 0 else 0

    D = np.zeros(npop)
    for i in range(npop):
        row = dist[i, :].copy()
        row.sort()
        idx = k_nn if (k_nn < row.shape[0]) else (row.shape[0] - 2)
        sigma_k = row[idx]
        if sigma_k <= 0.0 or not np.isfinite(sigma_k):
            sigma_k = 1e-10
        D[i] = 1.0 / (sigma_k + 2.0)

    F = R + D  # lower is better

    # ---- Median selection for the archive (never empty) ----
    tmp = F.copy()
    tmp.sort()
    med = tmp[npop // 2] if (npop % 2 == 1) else 0.5 * (tmp[npop // 2 - 1] + tmp[npop // 2])

    sel = np.zeros(npop, dtype=np.uint8)
    any_sel = False
    for i in range(npop):
        if F[i] <= med:
            sel[i] = 1
            any_sel = True
    if not any_sel:
        # keep the best by F
        best = 0
        bestv = F[0]
        for i in range(1, npop):
            if F[i] < bestv:
                bestv = F[i]
                best = i
        sel[best] = 1

    # Build archive arrays
    count = 0
    for i in range(npop):
        if sel[i] == 1:
            count += 1
    archive = np.empty((count, population.shape[1]))
    archive_obj = np.empty((count, nobj))
    archive_fit = np.empty(count)
    t = 0
    for i in range(npop):
        if sel[i] == 1:
            archive[t, :] = population[i, :]
            archive_obj[t, :] = objectives[i, :]
            archive_fit[t] = F[i]
            t += 1

    return archive, archive_fit, archive_obj


@njit
def crowd_distance(objectives, ranks):
    """
    NSGA-II crowding distance (used for selection tiebreak).

    Parameters
    ----------
    objectives : ndarray, shape (N, M)
        Objective values (minimization).
    ranks : ndarray, shape (N,)
        Rank labels per individual (0 = best).

    Returns
    -------
    distances : ndarray, shape (N,)
        Crowding distances; boundary points get +∞.
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


@njit
def tournament_selection_with_crowding(ranks, crowding_distances, pressure):
    """
    k-way tournament selection with crowding-distance tiebreak.

    Parameters
    ----------
    ranks : ndarray, shape (N,)
        Rank labels (0 = best).
    crowding_distances : ndarray, shape (N,)
        Crowding distances (higher = better diversity).
    pressure : int
        Tournament size (k >= 2).

    Returns
    -------
    selected_indices : ndarray, shape (N,)
        Indices of selected parents from the archive.
    """
    n_select = len(ranks)
    if pressure < 2:
        pressure = 2  # safety

    n_random = n_select * pressure
    n_perms = math.ceil(n_random / len(ranks))

    P = np.empty((n_random,), dtype=np.int32)
    for i in range(n_perms):
        P[i * len(ranks):(i + 1) * len(ranks)] = np.random.permutation(len(ranks))
    P = P[:n_random].reshape(n_select, pressure)

    selected_indices = np.full(n_select, -1, dtype=np.int32)
    for i in range(n_select):
        best = P[i, 0]
        for j in range(1, pressure):
            cand = P[i, j]
            if ranks[cand] < ranks[best]:
                best = cand
            elif ranks[cand] == ranks[best]:
                if crowding_distances[cand] > crowding_distances[best]:
                    best = cand
        selected_indices[i] = best

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
        Number of variables (D).
    crossover_prob : float
        Per-individual crossover probability.
    lower_bounds, upper_bounds : ndarray, shape (D,)
        Per-variable bounds to clamp offspring.

    Returns
    -------
    child_population : ndarray, shape (N, D)
        New population after crossover.
    """
    n_pop = population.shape[0]
    cross_probability = np.random.random(n_pop)
    do_cross = cross_probability < crossover_prob
    idxs = np.where(do_cross)[0]
    R = np.random.randint(0, n_pop, (idxs.shape[0], 2))
    child_population = population.copy()

    if num_vars > 1:
        cross_point = np.random.randint(1, num_vars, idxs.shape[0])
        for i in range(idxs.shape[0]):
            parent1, parent2 = R[i]
            point = cross_point[i]

            # Build child explicitly (Numba-friendly)
            child = np.empty(num_vars, dtype=population.dtype)
            for j in range(point):
                child[j] = population[parent1, j]
            for j in range(point, num_vars):
                child[j] = population[parent2, j]

            # Clamp to bounds
            for j in range(num_vars):
                if child[j] < lower_bounds[j]:
                    child[j] = lower_bounds[j]
                elif child[j] > upper_bounds[j]:
                    child[j] = upper_bounds[j]

            child_population[idxs[i], :] = child
    else:
        # D == 1 → average the parents, then clamp
        for i in range(idxs.shape[0]):
            parent1, parent2 = R[i]
            val = 0.5 * (population[parent1, 0] + population[parent2, 0])
            if val < lower_bounds[0]:
                val = lower_bounds[0]
            elif val > upper_bounds[0]:
                val = upper_bounds[0]
            child_population[idxs[i], 0] = val

    return child_population


@njit
def polynomial_mutation(population, mutation_rate, num_vars, lower_bounds, upper_bounds, eta_mut=20):
    """
    Polynomial mutation with bounds enforcement.

    Parameters
    ----------
    population : ndarray, shape (N, D)
        Input population to mutate.
    mutation_rate : float
        Per-variable mutation probability.
    num_vars : int
        Number of decision variables (D).
    lower_bounds, upper_bounds : ndarray, shape (D,)
        Bounds applied to mutated variables.
    eta_mut : float, optional
        Distribution index (larger = smaller steps). Default 20.

    Returns
    -------
    Y : ndarray, shape (N, D)
        Mutated population.
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


@njit
def euclidean_distances(X):
    """
    Dense Euclidean distance matrix.

    Parameters
    ----------
    X : ndarray, shape (N, D)
        Rows are points in D-dimensional space.

    Returns
    -------
    dist : ndarray, shape (N, N)
        Pairwise Euclidean distances. Diagonal is set to a large value to
        avoid degeneracy in nearest-neighbor queries.
    """
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e12)
    return dist
