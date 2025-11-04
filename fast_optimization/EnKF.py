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
    model_step,
    y_obs,
    t_indices,
    initialize_population,
    R,
    config: EnKFConfig = EnKFConfig(),
    *,
    model_step_batch=None,
    init_member_contexts=None,
):
    if config.rng_seed is not None:
        np.random.seed(config.rng_seed)

    y_obs = np.asarray(y_obs)
    if y_obs.ndim == 1:
        y_obs = y_obs[:, None]
    T, p = y_obs.shape
    Rm = _as_cov(R, p)

    pop, lb, ub = initialize_population(config.ensemble_size)
    if pop.shape[0] != config.ensemble_size:
        if pop.shape[0] < config.ensemble_size:
            reps = int(np.ceil(config.ensemble_size / pop.shape[0]))
            pop = np.vstack([pop] * reps)[:config.ensemble_size]
        else:
            pop = pop[:config.ensemble_size]
    N, D = pop.shape

    contexts = list(init_member_contexts) if init_member_contexts is not None else [None] * N

    theta_history = np.zeros((T, D))
    ensemble_history = np.zeros((T, D, N))
    innovations = np.full((T, p), np.nan)
    y_forecast_mean = np.full((T, p), np.nan)
    y_analysis_mean = np.full((T, p), np.nan)  # NEW: store analyzed shoreline mean

    for k, t_idx in enumerate(t_indices):
        pop = _apply_proc_noise(pop, config.parameter_process_std)
        pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)

        if config.use_batch_step and model_step_batch is not None:
            out = model_step_batch(pop, t_idx, contexts)
            if isinstance(out, tuple):
                Y_pred, new_contexts = out
                contexts = list(new_contexts)
            else:
                Y_pred = out
            Y_pred = np.asarray(Y_pred)
            if Y_pred.ndim == 1:
                Y_pred = Y_pred[:, None]
            if Y_pred.shape == (N, p):
                Yf = Y_pred.T
            elif Y_pred.shape == (p, N):
                Yf = Y_pred
            else:
                raise ValueError(f"model_step_batch must return (N, p) or (p, N), got {Y_pred.shape}")
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

        Theta_f = pop.T
        if config.inflation != 1.0:
            th_mean_inf = np.mean(Theta_f, axis=1, keepdims=True)
            Theta_f = th_mean_inf + config.inflation * (Theta_f - th_mean_inf)

        y_mean = np.mean(Yf, axis=1, keepdims=True)
        y_forecast_mean[k] = y_mean.ravel()
        if k == 0 and config.verbose:
            print(f"[EnKF] step 1: forecast_mean={y_mean.ravel()[0]:.3f}, obs={y_obs[0,0]:.3f}")

        X_f = np.vstack([Theta_f, Yf])      # (D+p, N)
        x_mean = np.mean(X_f, axis=1, keepdims=True)
        X_anom = X_f - x_mean

        obs_k = y_obs[k]
        valid = ~np.isnan(obs_k)
        if not np.any(valid):
            pop = Theta_f.T
            pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)
            # carry forecast state forward in contexts
            if p == 1:
                y_fore_members = Yf.T
                for j in range(N):
                    if contexts[j] is None: contexts[j] = {}
                    contexts[j]['y_old'] = float(y_fore_members[j, 0])
            theta_history[k] = np.mean(pop, axis=0)
            ensemble_history[k] = pop.T
            y_analysis_mean[k] = y_mean.ravel()  # NEW: analysis==forecast when no obs
            continue

        Yf_anom = X_anom[-p:, :]
        Yf_v = Yf_anom[valid, :]
        y_mean_v = y_mean[valid, :]
        obs_v = obs_k[valid]
        Rv = _as_cov(Rm[np.ix_(valid, valid)], int(np.sum(valid)))

        P_xy = (X_anom @ Yf_v.T) / (N - 1)
        P_yy = (Yf_v @ Yf_v.T) / (N - 1) + Rv
        P_yy = P_yy + np.eye(P_yy.shape[0]) * config.eps_jitter

        try:
            K = P_xy @ np.linalg.inv(P_yy)
        except np.linalg.LinAlgError:
            K = P_xy @ np.linalg.pinv(P_yy)

        innovations[k, valid] = obs_v - y_mean_v.ravel()

        if config.perturbed_observations:
            noise = np.random.multivariate_normal(mean=np.zeros(Rv.shape[0]), cov=Rv, size=N).T
            innov_members = (obs_v[:, None] + noise) - (y_mean_v + Yf_v)
        else:
            innov_members = (obs_v[:, None]) - (y_mean_v + Yf_v)

        X_a = X_f + K @ innov_members
        Theta_a = X_a[:-p, :]
        Y_a = X_a[-p:, :]

        pop = Theta_a.T
        pop = _enforce_bounds(pop, lb, ub, config.clip_to_bounds, config.reflect_bounds)

        # write analyzed shoreline to contexts and store mean
        if p == 1:
            y_a_members = Y_a.ravel()
            for j in range(N):
                if contexts[j] is None: contexts[j] = {}
                contexts[j]['y_old'] = float(y_a_members[j])
            y_analysis_mean[k, 0] = float(np.mean(y_a_members))   # NEW: mean analyzed shoreline
        else:
            y_analysis_mean[k] = np.mean(Y_a, axis=1)             # NEW: vector case

        theta_history[k] = np.mean(pop, axis=0)
        ensemble_history[k] = pop.T

        if config.verbose and (k % max(1, (T // 10)) == 0):
            print(f"[EnKF] step {k+1}/{T}  ||  |innov|={np.linalg.norm(innovations[k, valid]):.3e}")

    theta_best = theta_history[-1].copy()
    return theta_best, theta_history, ensemble_history, innovations, y_forecast_mean, y_analysis_mean
