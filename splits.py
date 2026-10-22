"""Create and save the 5 stratified train/val/test splits per dataset (Fair Rule 1).

For each seed in SEEDS: 70% train, 10% validation, 20% test, keeping the same
share of events in each part (stratified on the event indicator). Splits are
saved as row-index lists to splits/<dataset>_seed<k>.json so that nobody
splits the data on their own; everyone loads the same files.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from data import load_dataset

SPLITS_DIR = Path(__file__).parent / "splits"
SEEDS = (0, 1, 2, 3, 4)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.10
TEST_FRAC = 0.20
DATASET_NAMES = ("metabric", "support", "gbsg")


def make_split(df, seed: int) -> dict:
    """70/10/20 train/val/test split, stratified on the event indicator."""
    idx = df.index.to_numpy()
    events = df["event"].to_numpy()

    train_idx, rest_idx, _, rest_events = train_test_split(
        idx, events, train_size=TRAIN_FRAC, stratify=events, random_state=seed,
    )
    # rest_idx holds val+test (30%); split it so val ends up at 10% and
    # test at 20% of the *original* data.
    val_frac_of_rest = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=val_frac_of_rest, stratify=rest_events,
        random_state=seed,
    )
    return {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "test": test_idx.tolist(),
    }


def split_path(dataset_name: str, seed: int) -> Path:
    return SPLITS_DIR / f"{dataset_name}_seed{seed}.json"


def save_split(dataset_name: str, seed: int, split: dict) -> Path:
    SPLITS_DIR.mkdir(exist_ok=True)
    path = split_path(dataset_name, seed)
    with open(path, "w") as f:
        json.dump(split, f)
    return path


def load_split(dataset_name: str, seed: int) -> dict:
    with open(split_path(dataset_name, seed)) as f:
        return json.load(f)


def make_all_splits(dataset_names=DATASET_NAMES) -> None:
    for name in dataset_names:
        df = load_dataset(name)
        for seed in SEEDS:
            split = make_split(df, seed)
            save_split(name, seed, split)
            print(
                f"saved {name} seed {seed}: "
                f"train={len(split['train'])} val={len(split['val'])} "
                f"test={len(split['test'])}"
            )


if __name__ == "__main__":
    make_all_splits()
