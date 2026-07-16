#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from training import train_one


def main():
    parser = argparse.ArgumentParser(description="Train one NeuMF or ReviewOnly model.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--dataset", required=True, choices=["instant", "office", "digital"])
    parser.add_argument("--model", required=True, choices=["neumf", "reviewonly"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    train_one(config, args.dataset, args.model, args.seed, args.device, args.overwrite)


if __name__ == "__main__":
    main()
