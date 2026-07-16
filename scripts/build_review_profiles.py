#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from data import load_split, profile_dir
from reproducibility import set_global_seed


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


@torch.no_grad()
def encode_reviews(texts, tokenizer, model, device, batch_size, max_length):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = model(**encoded)
        pooled = functional.normalize(mean_pool(output.last_hidden_state, encoded["attention_mask"]), p=2, dim=1)
        vectors.append(pooled.cpu().numpy().astype(np.float32))
        done = min(start + batch_size, len(texts))
        if start == 0 or done == len(texts) or done % (batch_size * 20) == 0:
            print(f"[encode] {done}/{len(texts)}", flush=True)
    return np.concatenate(vectors)


def aggregate_profiles(ids, embeddings, entity_count):
    sums = np.zeros((entity_count, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(entity_count, dtype=np.int64)
    np.add.at(sums, ids, embeddings)
    np.add.at(counts, ids, 1)
    global_profile = embeddings.mean(axis=0)
    global_profile /= max(float(np.linalg.norm(global_profile)), 1e-12)
    seen = counts > 0
    sums[seen] /= counts[seen, None]
    sums[seen] /= np.maximum(np.linalg.norm(sums[seen], axis=1, keepdims=True), 1e-12)
    sums[~seen] = global_profile
    return sums, counts


def main():
    parser = argparse.ArgumentParser(description="Build train-only SBERT user and item review profiles.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--dataset", required=True, choices=["instant", "office", "digital"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = args.dataset
    output_dir = profile_dir(config, dataset)
    user_path = output_dir / "user_profiles.npy"
    item_path = output_dir / "item_profiles.npy"
    if user_path.exists() and item_path.exists() and not args.overwrite:
        print(f"[skip] {dataset}: profiles exist", flush=True)
        return

    encoder = config.raw["encoders"]["review"]
    set_global_seed(42)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        encoder["model"], local_files_only=args.offline
    )
    model = AutoModel.from_pretrained(
        encoder["model"], local_files_only=args.offline
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    frame = load_split(config, dataset, "train")
    reviews = frame["reviews"].fillna("").astype(str).tolist()
    embeddings = encode_reviews(
        reviews, tokenizer, model, device, args.batch_size, int(encoder["max_length"])
    )
    spec = config.dataset(dataset)
    users, _ = aggregate_profiles(
        frame["user_id"].to_numpy(np.int64), embeddings, int(spec["user_count"])
    )
    items, _ = aggregate_profiles(
        frame["item_id"].to_numpy(np.int64), embeddings, int(spec["item_count"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(user_path, users)
    np.save(item_path, items)
    print(f"[saved] {dataset}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
