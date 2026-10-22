"""Shared evaluation: C-index, IBS, D-calibration (Fair Rules 5-6).

Every model (classic or deep) must produce a survival-probability table of
shape (n_test_patients, len(time_grid)) evaluated at the same time grid
before calling `evaluate_predictions`. This is the *only* place metrics are
computed, so neither side can score itself differently (Fair Rule 6: nobody
writes their own evaluation).
"""
from __future__ import annotations

import numpy as np
from SurvivalEVAL import SurvivalEvaluator

N_TIME_POINTS = 100


def make_time_grid(duration_train: np.ndarray, duration_test: np.ndarray) -> np.ndarray:
    """100 points from 0 to the largest observed time (Fair Rule 5).

    Capped at train's max observed time, not train/test combined: the IPCW
    censoring-distribution weights used below are estimated from train only,
    so they are undefined past train's last observed time. Evaluating there
    anyway raises inside SurvivalEVAL ("Censoring survival must be finite and
    positive..."). In practice this only changes the grid when a held-out
    split's own max duration happens to exceed train's (rare, but it does
    happen with random splits -- e.g. gbsg seed 3).
    """
    t_max = float(duration_train.max())
    return np.linspace(0.0, t_max, N_TIME_POINTS)


def _censor_at_train_horizon(
    duration: np.ndarray, event: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Administratively censor any subject observed past train's horizon.

    IPCW weights a held-out subject by the train-estimated censoring
    survival at that subject's *own* observed time, not just at the grid
    points -- so a held-out subject timed past train's max observed time has
    an undefined weight regardless of the grid. Nothing past that point is
    identifiable from train data anyway, so such subjects are treated as
    censored at tau (the standard fixed-horizon convention), rather than at
    their true, later time.
    """
    past_tau = duration > tau
    clipped_duration = np.minimum(duration, tau)
    clipped_event = np.where(past_tau, False, event.astype(bool))
    return clipped_duration, clipped_event


def evaluate_predictions(
    pred_survs: np.ndarray,
    time_grid: np.ndarray,
    duration_test: np.ndarray,
    event_test: np.ndarray,
    duration_train: np.ndarray,
    event_train: np.ndarray,
) -> dict:
    """Compute the three agreed measures for one (model, dataset, seed) run.

    pred_survs: array of shape (n_test, len(time_grid)) -- S(t) for each test
    patient at each point in `time_grid`.

    Returns a dict with keys: c_index, ibs, dcal_p.
    """
    tau = float(duration_train.max())
    duration_test, event_test = _censor_at_train_horizon(duration_test, event_test, tau)

    evaluator = SurvivalEvaluator(
        pred_survs,
        time_grid,
        duration_test,
        event_test,
        duration_train,
        event_train,
    )
    c_index, _, _ = evaluator.concordance_time_dependent(method="Antolini")
    ibs = evaluator.integrated_brier_score(IPCW_weighted=True)
    dcal_p, _ = evaluator.d_calibration(num_bins=10)
    return {
        "c_index": float(c_index),
        "ibs": float(ibs),
        "dcal_p": float(dcal_p),
    }
