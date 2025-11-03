# EnKF.py
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Sequence, Any, Dict

# ---------------------------------------------------------------------
# Contracts expected from your codebase:
#   initialize_population(n) -> (pop, lb, ub) with shapes:
#       pop: (n, D), lb: (D,), ub: (D,)
#   model_step(theta, t_idx, context=None) -> y_pred OR (y_pred, new_context)
#       y_pred shape must match one row of y_obs: (p,) or scalar
# Optional (faster): model_step_batch(pop, t_idx) -> Y_pred OR (Y_pred, new_contexts)
#       Y_pred shape (N, p)
# ---------------------------------------------------------------------

@dataclass
class EnKFConfig:
    ensemble_size: int = 100
    parameter_process_std: float | np.ndarray = 0.0  # scalar or (D,)
    inflation: float = 1.0                           # multiplicative inflation on param anomalies
    clip_to_bounds: bool = True                      # True=clip, False=do-nothing
    reflect_bounds: bool = False                     # if True, reflect instead of clip
    perturbed_observations: bool = True              # stochastic EnKF
    rng_seed: Optional[int] = None                   # reproducibility
    eps_jitter: float = 1e-9                         # jitter added to P_yy for stability
    use_batch_step: bool = False                     # use model_step_batch if provided
    verbose: bool = False                            # light logging

def _as_cov(R: np.ndarray | float, p: int) -> np.ndarray:
    """Coerce R to a (p x p) covariance matrix."""
    R = np.asarray(R)
    if R.ndim == 0:
        return np.eye(p) * float(R)
    if R.ndim == 1:
        return np.diag(R**2)          # treat vector as std
    return R

def _apply_proc_noise(pop: np.ndarray, proc: float | np.ndarray) -> np.ndarray:
    if np.isscalar(proc):
        if proc == 0.0:
            return pop
        return pop + np.random.normal(0.0, float(proc), size=pop.shape)
    proc = np.asarray(proc)
    if proc.ndim == 1:
        return pop + np.random.normal(0.0, 1.0, size=pop.shape) * proc[None, :]
    raise ValueError("parameter_process_std must be scalar or (D,) vector")

def _enforce_bounds(pop: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                    clip: bool, reflect: bool) -> np.ndarray:
    if not clip and not reflect:
        return pop
    if reflect:
        # Reflect out-of-bounds components back into [lb, ub]
        width = ub - lb
        # Avoid zero-width dimensions
        width = np.where(width <= 0, 1.0, width)
        # Map to [0, width], reflect, then back to [lb, ub]
        z = (pop - lb) % (2 * width)
        pop = lb + np.where(z <= width, z, 2 * width - z)
        return pop
    # Default: clip
    return np.minimum(np.maximum(pop, lb), ub)

def enkf_parameter_assimilation(
    model_step: Callable[[np.ndarray, int, Optional[Any]], Any],
    y_obs: np.ndarray,
    t_indices: Sequence[int],
    initialize_population: Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    R: np.ndarray | float,
    config: EnKFConfig = EnKFConfig(),
    *,
    model_step_batch: Optional[Callable[[np.ndarray, int, Optional[Sequence[Any]]], Any]] = None,
    init_member_contexts: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble Kalman Filter for PARAMETER assimilation (random-walk parameters).

    Parameters
    ----------
    model_step : callable
        model_step(theta, t_idx, context=None) -> y_pred OR (y_pred, new_context).
        theta: (D,), y_pred: (p,) or scalar. If a context is used, return the updated context too.
    y_obs : ndarray, shape (T, p) or (T,)
        Observations at T assimilation steps (NaNs allowed -> update skipped, forecast still runs).
    t_indices : sequence of int
        Time indices consumed by your model. Your model interprets each t_idx and uses the
        appropriate forcing window internally.
    initialize_population : callable
        f(n) -> (pop, lb, ub). pop: (n,D), lb/ub: (D,).
    R : float or (p,) or (p,p)
        Observation error variance, std vector, or covariance matrix.
    config : EnKFConfig
        Filter hyperparameters.
    model_step_batch : optional callable
        model_step_batch(pop, t_idx, contexts=None) -> Y_pred OR (Y_pred, new_contexts),
        with Y_pred shape (N, p). Ignored if config.use_batch_step is False.
    init_member_contexts : optional sequence[Any]
        Initial per-member contexts (length N). If provided, passed to model_step(_batch) and updated each step.

    Returns
    -------
    theta_best : (D,)
        Final analysis ensemble mean.
    theta_history : (T, D)
        Analysis means per step.
    ensemble_history : (T, D, N)
        Analysis ensembles per step (transposed for convenient plotting).
    innovations : (T, p)
        y_obs - y_forecast_mean per step (NaN where obs missing).
    y_forecast_mean : (T, p)
        Ensemble mean of forecasted observations prior to analysis.

    Notes
    -----
    • Parameter-only EnKF (no latent state). We add small random-walk noise to θ if configured.
    • If your observations at step k contain NaNs, we skip the update for those components (or all if all NaN).
    • If you can vectorize your model with model_step_batch, set config.use_batch_step=True for speed.
    """
    if config.rng_seed is not None:
        np.random.seed(config.rng_seed)

    y_obs = np.asarray(y_obs)
    if y_obs.ndim == 1:
        y_obs = y_obs[:, None]
    T, p = y_obs.shape
    Rm = _as_cov(R, p)

    # Init ensemble & bounds
    pop, lb, ub = initialize_population(config.ensemble_size)
    if pop.shape[0] != config.ensemble_size:
        # enforce requested size
        if pop.shape[0] < config.ensemble_size:
            reps = int(np.ceil(config.ensemble_size / pop.shape[0]))
            pop = np.vstack([pop] * reps)[:config.ensemble_size]
        else:
            pop = pop[:config.ensemble_size]
    N, D = pop.shape

    # Optional per-member contexts
    contexts = list(init_member_contexts) if init_member_contexts is not None else [None] * N

    # Bookkeeping
    theta_history = np.zeros((T, D))
    ensemble_history = np.zeros((T, D, N))
    innovations = np.full((T, p), np.nan)
    y_forecast_mean = np.full((T, p), np.nan)

    for k, t_idx in enumerate(t_indices):
        # 1) Forecast: parameter random walk
        pop = _apply_proc_noise(pop, config.parameter_process_std)
        pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)

        # 2) Model forecast: Yf shape (p, N)
        if config.use_batch_step and model_step_batch is not None:
            out = model_step_batch(pop, t_idx, contexts)
            if isinstance(out, tuple):
                Y_pred, new_contexts = out
                contexts = list(new_contexts)
            else:
                Y_pred = out
            if Y_pred.ndim == 1:
                Y_pred = Y_pred[:, None]
            if Y_pred.shape[0] == N:
                Y_pred = Y_pred  # (N, p)
            else:
                # Expect (N, p). If (p, N) flip heuristically:
                if Y_pred.shape[1] == N and Y_pred.shape[0] == p:
                    Y_pred = Y_pred.T
            if Y_pred.shape != (N, p):
                raise ValueError(f"model_step_batch must return (N, p), got {Y_pred.shape}")
            Yf = Y_pred.T  # (p, N)
        else:
            Yf = np.zeros((p, N))
            for j in range(N):
                out = model_step(pop[j], t_idx, contexts[j])
                if isinstance(out, tuple):
                    yj, contexts[j] = out
                else:
                    yj = out
                yj = np.asarray(yj).ravel()
                if yj.size != p:
                    raise ValueError(f"model_step returned length {yj.size}, expected {p}")
                Yf[:, j] = yj

        # Optional inflation on parameter anomalies (pre-analysis)
        Theta_f = pop.T  # (D, N)
        if config.inflation != 1.0:
            theta_mean = np.mean(Theta_f, axis=1, keepdims=True)
            Theta_f = theta_mean + config.inflation * (Theta_f - theta_mean)

        # 3) Sample stats
        y_mean = np.mean(Yf, axis=1, keepdims=True)         # (p,1)
        theta_mean = np.mean(Theta_f, axis=1, keepdims=True)# (D,1)
        Yf_anom = Yf - y_mean                               # (p,N)
        Theta_anom = Theta_f - theta_mean                   # (D,N)
        y_forecast_mean[k] = y_mean.ravel()

        # 4) Observation mask (handle missing obs)
        obs_k = y_obs[k]  # (p,)
        valid = ~np.isnan(obs_k)
        if not np.any(valid):
            # No update; analysis == forecast
            pop = Theta_f.T
            pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)
            theta_mean_after = np.mean(pop, axis=0)
            theta_history[k] = theta_mean_after
            ensemble_history[k] = pop.T
            continue

        # Restrict to observed components
        Yf_v = Yf_anom[valid, :]                   # (p_v, N)
        y_mean_v = y_mean[valid, :]                # (p_v, 1)
        obs_v = obs_k[valid]                       # (p_v,)
        Rv = _as_cov(Rm[np.ix_(valid, valid)], np.sum(valid))

        P_xy = (Theta_anom @ Yf_v.T) / (N - 1)     # (D, p_v)
        P_yy = (Yf_v @ Yf_v.T) / (N - 1) + Rv      # (p_v, p_v)
        # jitter for stability
        P_yy = P_yy + np.eye(P_yy.shape[0]) * config.eps_jitter

        try:
            K = P_xy @ np.linalg.inv(P_yy)         # (D, p_v)
        except np.linalg.LinAlgError:
            K = P_xy @ np.linalg.pinv(P_yy)

        # 5) Innovations
        innovations[k, valid] = obs_v - y_mean_v.ravel()

        if config.perturbed_observations:
            noise = np.random.multivariate_normal(mean=np.zeros(Rv.shape[0]), cov=Rv, size=N).T  # (p_v, N)
            innov_members = (obs_v[:, None] + noise) - (y_mean_v + Yf_v)   # (p_v, N)
        else:
            innov_members = (obs_v[:, None] - (y_mean_v + Yf_v))           # (p_v, N)

        # 6) Analysis
        Theta_a = Theta_f + K @ innov_members   # (D, N)
        pop = Theta_a.T                         # (N, D)
        pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)

        # 7) Bookkeeping
        theta_history[k] = np.mean(pop, axis=0)
        ensemble_history[k] = pop.T

        if config.verbose and (k % max(1, (T // 10)) == 0):
            print(f"[EnKF] step {k+1}/{T}  ||  |innov|={np.linalg.norm(innovations[k, valid]):.3e}")

    theta_best = theta_history[-1].copy()
    return theta_best, theta_history, ensemble_history, innovations, y_forecast_mean
