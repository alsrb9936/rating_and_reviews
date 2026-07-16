#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias import attach_heldout_residuals, compute_user_bias
from config import load_config, parse_csv


def load_sentiment(config, dataset, split):
    path = config.root / "artifacts" / "sentiment" / dataset / f"{split}.csv"
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Compute leakage-free train-only rating and review-tone user biases.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--datasets", default="instant,office,digital")
    args = parser.parse_args()

    config = load_config(args.config)
    bias_cfg = config.raw["bias"]
    output_root = config.root / "results" / "derived"
    output_root.mkdir(parents=True, exist_ok=True)
    all_users = []
    all_heldout = []
    for dataset in parse_csv(args.datasets):
        train = load_sentiment(config, dataset, "train")
        users, references = compute_user_bias(
            train,
            minimum_interactions=int(bias_cfg["minimum_training_interactions"]),
            shrinkage_lambda=float(bias_cfg["shrinkage_lambda"]),
            tau=float(bias_cfg["case_threshold_tau"]),
        )
        users.insert(0, "dataset", dataset)
        all_users.append(users)
        for split in ("val", "test"):
            heldout = attach_heldout_residuals(load_sentiment(config, dataset, split), references)
            heldout.insert(0, "dataset", dataset)
            heldout.insert(1, "split", split)
            all_heldout.append(heldout)
    user_output = pd.concat(all_users, ignore_index=True)
    heldout_output = pd.concat(all_heldout, ignore_index=True)
    user_output.to_csv(output_root / "user_bias.csv", index=False)
    heldout_output.to_csv(output_root / "heldout_residuals.csv", index=False)
    print(f"[saved] {output_root / 'user_bias.csv'} rows={len(user_output)}")
    print(f"[saved] {output_root / 'heldout_residuals.csv'} rows={len(heldout_output)}")


if __name__ == "__main__":
    main()
