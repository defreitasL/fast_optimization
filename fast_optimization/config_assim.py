# config_assim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .EnKF import EnKFConfig, enkf_parameter_assimilation

Array = np.ndarray


def _coerce_R(R: Array | float, p: int) -> Array:
    """
    Accepts:
      - scalar variance (float)
      - vector of standard deviations (p,)
      - covariance matrix (p, p)
    Returns a (p, p) covariance matrix.
    """
    R = np.asarray(R)
    if R.ndim == 0:
        return np.eye(p) * float(R)
    if R.ndim == 1:
        if R.shape[0] != p:
            raise ValueError(f"R std-vector length {R.shape[0]} != obs dimension p={p}")
        return np.diag(R**2)
    if R.shape != (p, p):
        raise ValueError(f"R covariance shape {R.shape} != (p, p) with p={p}")
    return R


@dataclass
class ConfigAssim:
    """
    Thin EnKF runner (like your `config_cal`) that pulls from `model`:
      • model.model_step
      • model.t_indices
      • model.y_obs
      • model.initialize_population

    The observation error `R` is read from `cfg["R"]` (preferred). If not present,
    we fall back to `model.R`. If neither is available, a clear error is raised.

    Optional:
      • cfg["use_batch_step"]=True and either pass `model_step_batch=` here
        or define `model.model_step_batch`.
      • `init_member_contexts` if your model_step uses per-member contexts.
    """

    cfg: Dict[str, Any]
    enkf_cfg: EnKFConfig = None  # set in __post_init__

    def __post_init__(self) -> None:
        self.enkf_cfg = EnKFConfig(
            ensemble_size=int(self.cfg.get("ensemble_size", 100)),
            parameter_process_std=self.cfg.get("parameter_process_std", 0.0),
            inflation=float(self.cfg.get("inflation", 1.0)),
            clip_to_bounds=bool(self.cfg.get("clip_to_bounds", True)),
            reflect_bounds=bool(self.cfg.get("reflect_bounds", False)),
            perturbed_observations=bool(self.cfg.get("perturbed_observations", True)),
            rng_seed=self.cfg.get("rng_seed", None),
            eps_jitter=float(self.cfg.get("eps_jitter", 1e-9)),
            use_batch_step=bool(self.cfg.get("use_batch_step", False)),
            verbose=bool(self.cfg.get("verbose", False)),
        )

    def assimilate(
        self,
        model: Any
    ) -> Dict[str, Array]:
        """
        Run parameter EnKF using members found on `model` + R from cfg (or model).

        Required `model` attributes
        ---------------------------
        • model_step(theta, t_idx, context=None) -> y_pred OR (y_pred, new_context)
        • t_indices : Sequence[int]
        • y_obs : (T, p) or (T,)
        • initialize_population(n) -> (pop:(n,D), lb:(D,), ub:(D,))

        Optional `model` attributes
        ---------------------------
        • R : float or (p,) or (p,p)  (used if cfg["R"] is absent)
        • model_step_batch(pop, t_idx, contexts=None) -> Y_pred or (Y_pred, new_contexts)

        Returns
        -------
        dict with:
          "theta_best"       : (D,)
          "theta_history"    : (T, D)
          "ensemble_history" : (T, D, N)
          "innovations"      : (T, p)
          "y_forecast_mean"  : (T, p)
        """
        # ---- Pull required pieces from model ----
        for attr in ("model_step", "idx_assim", "Obs_splited", "init_par"):
            if not hasattr(model, attr):
                raise AttributeError(f"Model must provide '{attr}'")

        model_step = model.model_step
        model_step_batch = model.model_step_batch
        t_indices = model.idx_assim
        y_obs = model.Obs_splited[1:]
        initialize_population = model.init_par

        # ---- Determine R from cfg (preferred) or model ----
        R = self.cfg.get("R", None)

        if R is None:
            raise ValueError("Observation error 'R' must be provided in cfg['R'] or as model.R")

        # Ensure R matches the observation dimension
        if y_obs.ndim == 1:
            p = 1
        else:
            p = y_obs.shape[1]
        R = _coerce_R(R, p)

        # ---- Batch step (optional) ----
        if self.enkf_cfg.use_batch_step and model_step_batch is None:
            model_step_batch = getattr(model, "model_step_batch", None)
            if model_step_batch is None and self.enkf_cfg.verbose:
                print("[EnKF] use_batch_step=True but no batch step provided; falling back to per-member calls.")

        # ---- Run core EnKF ----
        theta_best, theta_hist, ens_hist, innov, y_fore_mean, y_anal_mean = enkf_parameter_assimilation(
            model_step=model_step,
            y_obs=y_obs,
            t_indices=t_indices,
            initialize_population=initialize_population,
            R=R,
            config=self.enkf_cfg,
            model_step_batch=model_step_batch,
            init_member_contexts=[{'y_old': float(model.Yini)} for _ in range(self.enkf_cfg.ensemble_size)],
        )

        res = {
                "theta_best": theta_best,
                "theta_history": theta_hist,
                "ensemble_history": ens_hist,
                "innovations": innov,
                "y_forecast_mean": y_fore_mean,
                "y_analysis_mean": y_anal_mean,   # NEW
            }
        return res