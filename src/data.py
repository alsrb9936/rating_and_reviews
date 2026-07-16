import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLIT_FILE = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
def split_path(config, dataset, split):
    return config.root / "data" / dataset / SPLIT_FILE[split]


def profile_dir(config, dataset):
    return config.root / "data" / dataset / "profiles"


def load_split(config, dataset, split):
    return pd.read_csv(split_path(config, dataset, split))


class RatingDataset(Dataset):
    def __init__(self, config, dataset, split):
        frame = load_split(config, dataset, split)
        self.user_ids = torch.as_tensor(frame["user_id"].to_numpy(np.int64), dtype=torch.long)
        self.item_ids = torch.as_tensor(frame["item_id"].to_numpy(np.int64), dtype=torch.long)
        self.ratings = torch.as_tensor(frame["ratings"].to_numpy(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):
        return self.user_ids[index], self.item_ids[index], self.ratings[index]


def load_profiles(config, dataset):
    root = profile_dir(config, dataset)
    return np.load(root / "user_profiles.npy"), np.load(root / "item_profiles.npy")
