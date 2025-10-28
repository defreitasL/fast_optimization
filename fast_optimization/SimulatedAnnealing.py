import numpy as np
import copy
from .objectives_functions import multi_obj_func
from .metrics import backtot

import copy
import numpy as np


def simulated_annealing(
    model_simulation,
    Obs,
    initialize_population,
    max_iterations,
    initial_temperature,
    cooling_rate,
    index_metrics,
    n_restarts=5,
):
    """
    Simulated Annealing (SA) for single-solution optimization with multi-objective
    support via simple scalarization (sum of minimized objectives).

    The function assumes `multi_obj_func(Obs, sim, index_metrics)` returns an
    objective vector in MINIMIZATION sense (as per your pipeline where metrics that
    should be maximized are internally flipped, e.g., 1 - metric).

    Strategy
    --------
    • Single-solution annealing per restart (no population).
    • Proposal: uniform perturbation in a fraction of the parameter range, then clamped to bounds.
    • Energy (to minimize): sum of the objective vector.
    • Acceptance: classic SA rule; always accept if Δ <= 0, else accept with
      probability exp(-Δ / T).
    • Cooling: geometric schedule with rate `cooling_rate`, plus a small floor to
      avoid numerical issues.

    Parameters
    ----------
    model_simulation : Callable[[ndarray], Any]
        Function that runs the forward model for a single parameter vector.
        Output must be consumable by `multi_obj_func`.
    Obs : Any
        Observations passed through to `multi_obj_func` to compute objective values.
    initialize_population : Callable[[int], Tuple[ndarray, ndarray, ndarray]]
        Function returning `(population, lower_bounds, upper_bounds)` where:
          - population: (N, D) initial parameter vectors,
          - lower_bounds, upper_bounds: (D,) bounds arrays.
        SA uses the first individual: `population[0]`.
    max_iterations : int
        Maximum SA iterations per restart.
    initial_temperature : float
        Starting temperature T0.
    cooling_rate : float
        Multiplicative cooling factor per iteration (e.g., 0.98).
    index_metrics : Sequence[int]
        Indices selecting which objectives from `multi_obj_func` are active.
    n_restarts : int, optional
        Number of independent SA restarts; best across restarts is returned.

    Returns
    -------
    best_solution : ndarray, shape (D,)
        Best parameter vector found across all restarts.
    best_fitness : list[float]
        Objective vector (minimization sense) for `best_solution`.
    fitness_history : list[list[float]]
        History of the best objective vector over iterations (from the *last* restart).

    Notes
    -----
    • Printing of metric values at intervals uses your `backtot()` mask semantics:
      if a metric is “minimize” it prints directly; if it is “maximize” in the
      original sense, you can print `1 - obj` to show the natural metric value.
      Here we print the first (and only) metric, consistent with your earlier code.
    """

    print(f"Starting Simulated Annealing optimization with {n_restarts} restarts...")

    # Retrieve metric naming/mask. `mask[0] == True` means the metric is already
    # in minimization sense; `False` means originally a maximization metric.
    metrics_name_list, mask_list = backtot()
    metric_name = [metrics_name_list[k] for k in index_metrics][0]
    is_minimize_metric = [mask_list[k] for k in index_metrics][0]

    best_solution = None
    best_fitness = None  # numpy array (M,)

    # Logging cadence (safe for small max_iterations)
    log_every = max(1, max_iterations // 10)

    for restart in range(n_restarts):
        print(f"Starting {restart + 1}/{n_restarts}")

        # --------------------------------------------------------------
        # Initialization (single solution)
        # --------------------------------------------------------------
        population, lower_bounds, upper_bounds = initialize_population(1)
        current_solution = population[0]
        num_params = current_solution.shape[0]

        # Evaluate initial solution
        current_simulation = model_simulation(current_solution)
        current_fitness = np.asarray(
            multi_obj_func(Obs, current_simulation, index_metrics), dtype=float
        )  # shape (M,)
        current_energy = float(np.sum(current_fitness))  # scalar

        # Initialize global best across restarts
        if best_solution is None:
            best_solution = copy.deepcopy(current_solution)
            best_fitness = current_fitness.copy()
            best_energy = current_energy
        else:
            if current_energy < best_energy:
                best_solution = copy.deepcopy(current_solution)
                best_fitness = current_fitness.copy()
                best_energy = current_energy

        # Per-restart tracking (we return the last restart's history)
        fitness_history = [best_fitness.tolist()]

        # Annealing temperature
        temperature = float(initial_temperature)
        T_floor = 1e-12  # avoid divide-by-zero/underflow

        # --------------------------------------------------------------
        # Main SA loop
        # --------------------------------------------------------------
        for iteration in range(max_iterations):
            # 1) Propose a neighbor (uniform local step, clamped to bounds)
            step = np.random.uniform(-0.1, 0.1, size=num_params) * (upper_bounds - lower_bounds)
            candidate_solution = current_solution + step
            candidate_solution = np.clip(candidate_solution, lower_bounds, upper_bounds)

            # 2) Evaluate candidate
            candidate_simulation = model_simulation(candidate_solution)
            candidate_fitness = np.asarray(
                multi_obj_func(Obs, candidate_simulation, index_metrics), dtype=float
            )
            candidate_energy = float(np.sum(candidate_fitness))

            # 3) Acceptance test (Metropolis rule on scalarized energy)
            delta = candidate_energy - current_energy
            if delta <= 0.0:
                accept = True
            else:
                # Temperature floor to keep exp stable
                accept_prob = np.exp(-delta / max(temperature, T_floor))
                accept = np.random.rand() < accept_prob

            if accept:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                current_energy = candidate_energy

            # 4) Update the global best if improved
            if current_energy < best_energy:
                best_solution = copy.deepcopy(current_solution)
                best_fitness = current_fitness.copy()
                best_energy = current_energy

            # 5) Track history (best-so-far)
            fitness_history.append(best_fitness.tolist())

            # 6) Cool down
            temperature = max(temperature * cooling_rate, T_floor)

            # 7) Progress log (prints the first metric in human sense)
            if iteration % log_every == 0:
                if is_minimize_metric:
                    human_val = best_fitness[0]
                else:
                    # original metric was a maximization; invert for display
                    human_val = 1.0 - best_fitness[0]
                print(
                    f"Iteration {iteration}/{max_iterations}, "
                    f"{metric_name}: {human_val:.3f}, Temperature: {temperature:.3e}"
                )

    return best_solution, best_fitness.tolist(), fitness_history
