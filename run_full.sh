#!/usr/bin/env bash
set -euo pipefail

DATASET=${1:-instant}
SEED=${2:-7}
DEVICE=${3:-cuda:0}

python scripts/build_review_profiles.py --dataset "$DATASET" --device "$DEVICE"
python scripts/extract_sentiment.py --dataset "$DATASET" --split train --device "$DEVICE"
python scripts/extract_sentiment.py --dataset "$DATASET" --split val --device "$DEVICE"
python scripts/extract_sentiment.py --dataset "$DATASET" --split test --device "$DEVICE"
python scripts/train.py --dataset "$DATASET" --model neumf --seed "$SEED" --device "$DEVICE"
python scripts/train.py --dataset "$DATASET" --model reviewonly --seed "$SEED" --device "$DEVICE"
python scripts/save_predictions.py --dataset "$DATASET" --seed "$SEED" --device "$DEVICE"
