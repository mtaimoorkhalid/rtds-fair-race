"""Build the results table and run the Wilcoxon signed-rank tests (Fair Rule 7).

Reads every CSV under results/classic/ and results/deep/ (Fair Rule: nobody
writes their own evaluation -- these CSVs all came from evaluate.py), builds
one results table (mean +/- std per model per dataset across the 5 seeds),
and runs a paired Wilcoxon signed-rank test on the 15 (dataset, seed) pairs
for the main comparison: each deep model vs. RandomSurvivalForest. If more
than one pair is compared, a Bonferroni correction is applied (L8).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

RESULTS_DIR = Path(__file__).parent / "results"
METRICS = ("c_index", "ibs", "dcal_p")
CLASSIC_MODELS = {"coxph", "rsf", "gbsa"}  # the config keys used as model names


def load_all_results() -> pd.DataFrame:
    """Load every results/<side>/*.csv into one long DataFrame."""
    frames = []
    for side in ("classic", "deep"):
        side_dir = RESULTS_DIR / side
        if not side_dir.exists():
            continue
        for csv_path in sorted(side_dir.glob("*.csv")):
            frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(
            f"No results CSVs found under {RESULTS_DIR}/classic or {RESULTS_DIR}/deep"
        )
    return pd.concat(frames, ignore_index=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of each metric, per model per dataset, across the 5 seeds."""
    agg = df.groupby(["model", "dataset"])[list(METRICS)].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


def paired_wilcoxon(df: pd.DataFrame, model_a: str, model_b: str, metric: str = "c_index"):
    """Paired Wilcoxon signed-rank test on the shared (dataset, seed) pairs.

    Returns (statistic, p_value, n_pairs), or None if the two models don't
    share any (dataset, seed) pairs yet.
    """
    a = df[df["model"] == model_a].set_index(["dataset", "seed"])[metric]
    b = df[df["model"] == model_b].set_index(["dataset", "seed"])[metric]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    stat, p = wilcoxon(a.loc[common], b.loc[common])
    return stat, p, len(common)


def bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni-corrected p-values (L8: correct when comparing more than one pair)."""
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def main():
    df = load_all_results()
    table = summary_table(df)
    print("=== Results table (mean +/- std across 5 seeds) ===")
    print(table.to_string(index=False))

    out_path = RESULTS_DIR / "summary_table.csv"
    table.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    models = sorted(df["model"].unique())
    print(f"\nmodels found: {models}")

    if "rsf" not in models:
        print("rsf results not found yet -- run classic/run_classic.py first.")
        return

    deep_models = [m for m in models if m not in CLASSIC_MODELS]
    if not deep_models:
        print("No deep-model results found yet -- waiting on deep/run_deep.py.")
        return

    print("\n=== Wilcoxon: each deep model vs rsf (15 pairs) ===")
    p_values, labels = [], []
    for metric in ("c_index", "ibs"):
        for dm in deep_models:
            result = paired_wilcoxon(df, dm, "rsf", metric=metric)
            if result is None:
                print(f"  {dm} vs RSF ({metric}): no overlapping (dataset, seed) pairs yet")
                continue
            stat, p, n = result
            a = df[df["model"] == dm].set_index(["dataset", "seed"])[metric]
            b = df[df["model"] == "rsf"].set_index(["dataset", "seed"])[metric]
            diff = (a - b).dropna()
            print(f"  {dm} vs RSF ({metric}): W={stat:.3f}, p={p:.4f}, "
                  f"n_pairs={n}, mean diff={diff.mean():+.4f}")
            p_values.append(p)
            labels.append(f"{dm} vs RSF ({metric})")

    if len(p_values) > 1:
        corrected = bonferroni(p_values)
        print(f"\n  Bonferroni-corrected over {len(p_values)} tests (L8):")
        for label, p_raw, p_corr in zip(labels, p_values, corrected):
            print(f"    {label}: p_raw={p_raw:.4f} -> p_corrected={p_corr:.4f}")


if __name__ == "__main__":
    main()