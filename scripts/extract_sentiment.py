#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from data import load_split


def main():
    parser = argparse.ArgumentParser(description="Extract sentiment for one dataset split.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--dataset", required=True, choices=["instant", "office", "digital"])
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output = config.root / "artifacts" / "sentiment" / args.dataset / f"{args.split}.csv"
    if output.exists() and not args.overwrite:
        print(f"[skip] {output}")
        return

    encoder = config.raw["encoders"]["sentiment"]
    labels = list(encoder["labels"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(encoder["model"], local_files_only=args.offline)
    model = AutoModelForSequenceClassification.from_pretrained(
        encoder["model"], local_files_only=args.offline
    ).to(device).eval()

    model_labels = [str(model.config.id2label[index]).strip().lower().replace("_", " ") for index in range(5)]
    reorder = [model_labels.index(label) for label in labels] if set(model_labels) == set(labels) else list(range(5))

    frame = load_split(config, args.dataset, args.split)
    texts = frame["reviews"].fillna("").astype(str).tolist()
    batches = []
    for start in range(0, len(texts), args.batch_size):
        encoded = tokenizer(
            texts[start : start + args.batch_size],
            return_tensors="pt",
            truncation=True,
            max_length=int(encoder["max_length"]),
            padding=True,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[:, reorder]
        batches.append(probabilities.cpu().numpy())
        print(f"[sentiment] {min(start + args.batch_size, len(texts))}/{len(texts)}", flush=True)

    probabilities = np.concatenate(batches)
    hard_labels = probabilities.argmax(axis=1)
    result = pd.DataFrame(
        {
            "row_id": np.arange(len(frame)),
            "user_id": frame["user_id"].to_numpy(),
            "item_id": frame["item_id"].to_numpy(),
            "rating": frame["ratings"].to_numpy(float),
            "review_length": [len(re.findall(r"[A-Za-z0-9']+", text)) for text in texts],
            "sentiment_label": [labels[index] for index in hard_labels],
            "sentiment_1_5": hard_labels + 1,
            "sentiment_expected_1_5": probabilities @ np.arange(1, 6, dtype=float),
        }
    )
    for index in range(5):
        result[f"sentiment_p{index + 1}"] = probabilities[:, index]

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"[saved] {output} rows={len(result)}")


if __name__ == "__main__":
    main()
