# RecSys 2026 User Calibration

This repository trains NeuMF and ReviewOnly, saves predictions from both models for the same test interactions, and analyzes user-level rating-review calibration.

## Project Structure

```text
configs/       Training and bias settings
src/           Models, training, data loading, and bias calculation
scripts/       Profile building, sentiment extraction, training, and prediction export
analysis/      Bias and calibration analysis
data/          Fixed train/validation/test splits
artifacts/     Checkpoints, sentiment scores, and interaction-level predictions
results/       Analysis CSV files
```

## Training a Model

Specify the GPU directly when running a command. The project does not automatically distribute jobs across GPUs.

```bash
python scripts/train.py \
  --dataset instant \
  --model neumf \
  --seed 7 \
  --device cuda:0

python scripts/train.py \
  --dataset instant \
  --model reviewonly \
  --seed 7 \
  --device cuda:0
```

Training runs for up to 200 epochs with early stopping patience 10 based on validation MSE. Training-review profiles must be generated before training ReviewOnly.

```bash
python scripts/build_review_profiles.py --dataset instant --device cuda:0
```

## Saving Interaction-Level Predictions

After both models have been trained, run:

```bash
python scripts/save_predictions.py \
  --dataset instant \
  --seed 7 \
  --device cuda:0
```

The outputs are saved to `artifacts/predictions/instant/seed7/val.csv` and `test.csv`. Each row contains `row_id`, `user_id`, `item_id`, `rating`, `pred_neumf`, and `pred_reviewonly`.

To run profile generation, sentiment extraction, model training, and prediction export for one dataset and seed:

```bash
./run_full.sh instant 7 cuda:0
```

## Sentiment and Bias Analysis

Specify the dataset, split, and GPU directly when extracting sentiment scores.

```bash
python scripts/extract_sentiment.py --dataset instant --split train --device cuda:0
python scripts/extract_sentiment.py --dataset instant --split val --device cuda:0
python scripts/extract_sentiment.py --dataset instant --split test --device cuda:0
```

Aggregate the prepared results for all three datasets and ten seeds with:

```bash
./run_analysis.sh
```

The main outputs are:

- `results/derived/user_bias.csv`
- `results/derived/interaction_eval.csv`
- `results/tables/*.csv`

User biases are calculated from training interactions only. Validation and test review sentiment is used only to evaluate held-out bias stability and is never used as model input.
