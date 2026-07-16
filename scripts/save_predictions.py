#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from data import load_split
from training import predict_split


def main():
    parser = argparse.ArgumentParser(description="Save paired NeuMF and ReviewOnly interaction predictions.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--dataset", required=True, choices=["instant", "office", "digital"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = args.dataset
    seed = args.seed
    output_dir = config.root / "artifacts" / "predictions" / dataset / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        output = output_dir / f"{split}.csv"
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}", flush=True)
            continue
        frame = load_split(config, dataset, split)
        result = pd.DataFrame(
            {
                "row_id": range(len(frame)),
                "user_id": frame["user_id"].to_numpy(),
                "item_id": frame["item_id"].to_numpy(),
                "rating": frame["ratings"].to_numpy(float),
                "pred_neumf": predict_split(config, dataset, "neumf", seed, split, args.device),
                "pred_reviewonly": predict_split(config, dataset, "reviewonly", seed, split, args.device),
            }
        )
        result.to_csv(output, index=False)
        print(f"[saved] {output} rows={len(result)}", flush=True)


if __name__ == "__main__":
    main()
