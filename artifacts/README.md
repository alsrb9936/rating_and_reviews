# Generated artifacts

- `checkpoints/<dataset>/seed<seed>/<model>/best_model.pt`: selected model weights.
- `checkpoints/.../run.json`: best epoch and validation metrics.
- `predictions/<dataset>/seed<seed>/{val,test}.csv`: aligned NeuMF and ReviewOnly predictions.
- `sentiment/<dataset>/{train,val,test}.csv`: hard label, expected 1--5 score, and five class probabilities.
Prediction files have a stable key `(dataset, seed, split, row_id, user_id, item_id)`. `row_id` preserves fixed split order and must be unique within a split.
