import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import canonical_model_name
from data import RatingDataset, load_profiles
from models import NeuMF, ReviewOnly
from reproducibility import set_global_seed


def checkpoint_dir(config, dataset, seed, model_name):
    return config.root / "artifacts" / "checkpoints" / dataset / f"seed{seed}" / canonical_model_name(model_name)


def create_model(config, dataset, model_name):
    name = canonical_model_name(model_name)
    spec = config.dataset(dataset)
    hparams = config.model_hparams(name, dataset)
    if name == "neumf":
        dimension = int(config.model(name)["embedding_dimension"])
        return NeuMF(
            user_count=int(spec["user_count"]),
            item_count=int(spec["item_count"]),
            dropout=float(hparams["dropout"]),
            dimension=dimension,
        )
    user_profiles, item_profiles = load_profiles(config, dataset)
    return ReviewOnly(user_profiles, item_profiles, dropout=float(hparams["dropout"]))


def make_loader(
    config,
    dataset,
    split,
    batch_size,
    seed,
    shuffle,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        RatingDataset(config, dataset, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config.raw["training"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    count = 0
    for user_ids, item_ids, ratings in loader:
        user_ids = user_ids.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        ratings = ratings.to(device, non_blocking=True)
        predictions = model(user_ids, item_ids)
        difference = predictions - ratings
        squared_error += torch.sum(difference.square()).item()
        absolute_error += torch.sum(difference.abs()).item()
        count += ratings.numel()
    mse = squared_error / count
    return {"n": count, "mse": mse, "rmse": math.sqrt(mse), "mae": absolute_error / count}


def train_one(
    config,
    dataset,
    model_name,
    seed,
    device_name,
    overwrite=False,
):
    name = canonical_model_name(model_name)
    output_dir = checkpoint_dir(config, dataset, seed, name)
    model_path = output_dir / "best_model.pt"
    metadata_path = output_dir / "run.json"
    if model_path.exists() and metadata_path.exists() and not overwrite:
        print(f"[skip] {dataset}/seed{seed}/{name}: checkpoint exists", flush=True)
        return output_dir

    set_global_seed(seed, deterministic=bool(config.raw["training"]["deterministic_algorithms"]))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model = create_model(config, dataset, name).to(device)
    model_cfg = config.model_hparams(name, dataset)
    batch_size = int(model_cfg["batch_size"])
    train_loader = make_loader(config, dataset, "train", batch_size, seed, shuffle=True)
    val_loader = make_loader(config, dataset, "val", batch_size, seed, shuffle=False)

    train_cfg = config.raw["training"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(model_cfg["weight_decay"]),
    )
    scheduler_cfg = train_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_cfg["step_size"]),
        gamma=float(scheduler_cfg["gamma"]),
    )
    criterion = nn.MSELoss()
    best_mse = float("inf")
    epochs_without_improvement = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(int(train_cfg["epochs"])):
        started = time.time()
        model.train()
        loss_sum = 0.0
        count = 0
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            ratings = ratings.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * ratings.numel()
            count += ratings.numel()
        scheduler.step()

        train_mse = loss_sum / count
        validation = evaluate(model, val_loader, device)
        improved = validation["mse"] < best_mse - float(train_cfg["min_delta"])
        if improved:
            best_mse = validation["mse"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            payload = {
                "dataset": dataset,
                "model": name,
                "seed": seed,
                "best_epoch": epoch + 1,
                "train_mse": train_mse,
                "validation": validation,
                "model_hyperparameters": model_cfg,
            }
            metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            epochs_without_improvement += 1

        print(
            f"[epoch] {dataset}/seed{seed}/{name} epoch={epoch} "
            f"train_mse={train_mse:.6f} val_mse={validation['mse']:.6f} "
            f"best={best_mse:.6f} no_improve={epochs_without_improvement}/"
            f"{train_cfg['early_stopping_patience']} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        if epochs_without_improvement >= int(train_cfg["early_stopping_patience"]):
            break

    return output_dir


def load_trained_model(
    config, dataset, model_name, seed, device
):
    name = canonical_model_name(model_name)
    path = checkpoint_dir(config, dataset, seed, name) / "best_model.pt"
    model = create_model(config, dataset, name)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_split(
    config,
    dataset,
    model_name,
    seed,
    split,
    device_name,
):
    name = canonical_model_name(model_name)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = load_trained_model(config, dataset, name, seed, device)
    batch_size = int(config.model_hparams(name, dataset)["batch_size"])
    loader = make_loader(config, dataset, split, batch_size, seed, shuffle=False)
    outputs = []
    clamp = config.model(name)["test_prediction_clamp"]
    for user_ids, item_ids, _ in loader:
        predictions = model(user_ids.to(device), item_ids.to(device))
        if clamp is not None:
            predictions = predictions.clamp(float(clamp[0]), float(clamp[1]))
        outputs.append(predictions.cpu().numpy())
    return np.concatenate(outputs)
