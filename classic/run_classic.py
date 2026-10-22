"""Train + tune the classic models (Usama's side): CoxPH, RSF, (GBSA optional).

For every dataset x seed x model: 20 random tuning tries on train, scored on
validation; the best try is retrained and scored once on test. Results are
written to results/classic/<model>.csv with one row per (model, dataset,
seed), matching the shared format agreed in the project plan.

Tuning score: validation C-index (see README / project plan decision log).
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import load_dataset, preprocess_split  # noqa: E402
from evaluate import evaluate_predictions, make_time_grid  # noqa: E402
from splits import DATASET_NAMES, SEEDS, load_split  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "classic.yaml"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "classic"
TUNING_SCORE = "c_index"  # agreed tuning score: validation C-index

MODEL_CLASSES = {
    "CoxPHSurvivalAnalysis": CoxPHSurvivalAnalysis,
    "RandomSurvivalForest": RandomSurvivalForest,
    "GradientBoostingSurvivalAnalysis": GradientBoostingSurvivalAnalysis,
}


def hash_str(s: str) -> int:
    """Stable, process-independent hash (unlike Python's randomized hash())."""
    return zlib.crc32(s.encode("utf-8"))


def to_structured_y(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    """sksurv needs a structured array of (event: bool, time: float)."""
    return np.array(
        list(zip(event.astype(bool), duration)),
        dtype=[("event", bool), ("time", float)],
    )


def sample_params(param_spec: dict, rng: np.random.Generator) -> dict:
    """Draw one random hyperparameter configuration from the YAML spec."""
    sampled = {}
    for name, spec in param_spec.items():
        kind = spec["type"]
        if kind == "loguniform":
            sampled[name] = float(
                np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"])))
            )
        elif kind == "uniform":
            sampled[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif kind == "int":
            sampled[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif kind == "categorical":
            choices = spec["choices"]
            sampled[name] = choices[rng.integers(0, len(choices))]
        else:
            raise ValueError(f"Unknown param type '{kind}' for '{name}'")
    return sampled


def predict_surv_table(model, X: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """Survival probability table of shape (n_samples, len(time_grid)).

    sksurv's step functions only accept query times inside the domain seen
    during training (0 to the model's max training time). Query times beyond
    that are clipped to the domain max, i.e. held flat at the last estimated
    value -- the standard convention when extrapolating a survival curve.
    """
    surv_fns = model.predict_survival_function(X)
    rows = []
    for fn in surv_fns:
        t_max = fn.domain[1]
        clipped = np.minimum(time_grid, t_max)
        rows.append(fn(clipped))
    return np.vstack(rows)


def run_model_on_split(
    model_name: str,
    model_cfg: dict,
    dataset_name: str,
    seed: int,
) -> dict | None:
    df = load_dataset(dataset_name)
    split = load_split(dataset_name, seed)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_split(
        df, split["train"], split["val"], split["test"]
    )
    dur_train, ev_train = y_train
    dur_val, ev_val = y_val
    dur_test, ev_test = y_test

    y_train_struct = to_structured_y(dur_train, ev_train)
    model_cls = MODEL_CLASSES[model_cfg["class"]]
    n_trials = model_cfg["n_trials"]
    param_spec = model_cfg["params"]

    # Deterministic per (dataset, seed, model) random-search stream.
    # NOTE: do not use Python's built-in hash() here -- it is randomized per
    # process (PYTHONHASHSEED) unless explicitly fixed, so it silently breaks
    # reproducibility between runs. np.random.SeedSequence gives a stable,
    # well-mixed seed from the same inputs every time.
    seed_seq = np.random.SeedSequence([seed, hash_str(dataset_name), hash_str(model_name)])
    rng = np.random.default_rng(seed_seq)

    val_time_grid = make_time_grid(dur_train, dur_val)
    best_score = -np.inf
    best_params = None

    t0 = time.perf_counter()
    for trial in range(n_trials):
        params = sample_params(param_spec, rng)
        try:
            model = model_cls(**params)
            model.fit(X_train, y_train_struct)
            pred_val = predict_surv_table(model, X_val, val_time_grid)
            metrics = evaluate_predictions(
                pred_val, val_time_grid, dur_val, ev_val, dur_train, ev_train
            )
            score = metrics[TUNING_SCORE]
        except Exception as exc:  # a solver can fail silently/loudly (see lit review)
            print(f"  [{model_name}/{dataset_name}/seed{seed}] trial {trial} "
                  f"failed with params {params}: {exc}")
            continue
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        print(f"  [{model_name}/{dataset_name}/seed{seed}] all {n_trials} trials "
              f"failed -- skipping this run")
        return None

    # Retrain best config on train, evaluate once on test.
    final_model = model_cls(**best_params)
    final_model.fit(X_train, y_train_struct)
    train_seconds = time.perf_counter() - t0

    test_time_grid = make_time_grid(dur_train, dur_test)
    pred_test = predict_surv_table(final_model, X_test, test_time_grid)
    test_metrics = evaluate_predictions(
        pred_test, test_time_grid, dur_test, ev_test, dur_train, ev_train
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "seed": seed,
        "c_index": test_metrics["c_index"],
        "ibs": test_metrics["ibs"],
        "dcal_p": test_metrics["dcal_p"],
        "best_params": str(best_params),
        "train_seconds": train_seconds,
    }


def load_done_pairs(out_path: Path) -> set[tuple[str, int]]:
    """(dataset, seed) pairs already written to this model's CSV."""
    if not out_path.exists():
        return set()
    existing = pd.read_csv(out_path)
    return set(zip(existing["dataset"], existing["seed"].astype(int)))


def append_row(out_path: Path, row: dict) -> None:
    """Append one finished row to the CSV, writing a header only if new.

    Writing one row at a time -- instead of collecting everything in memory
    and writing once at the end -- means the run can be safely killed and
    resumed later (e.g. overnight): nothing already written needs to be
    recomputed, only the remaining (dataset, seed) pairs are run.
    """
    pd.DataFrame([row]).to_csv(
        out_path, mode="a", header=not out_path.exists(), index=False
    )


def main(dataset_names=DATASET_NAMES, seeds=SEEDS):
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_cfg in config["models"].items():
        if not model_cfg.get("enabled", True):
            print(f"skipping {model_name} (disabled in classic.yaml)")
            continue

        out_path = RESULTS_DIR / f"{model_name}.csv"
        done = load_done_pairs(out_path)

        for dataset_name in dataset_names:
            for seed in seeds:
                if (dataset_name, seed) in done:
                    print(f"skipping {model_name}/{dataset_name}/seed{seed} "
                          f"(already in {out_path.name})")
                    continue
                print(f"running {model_name} on {dataset_name} seed {seed} ...")
                row = run_model_on_split(model_name, model_cfg, dataset_name, seed)
                if row is not None:
                    append_row(out_path, row)

        n_rows = len(pd.read_csv(out_path)) if out_path.exists() else 0
        print(f"{out_path} now has {n_rows} rows")


if __name__ == "__main__":
    main()
