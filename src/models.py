import torch
from torch import nn


class NeuMF(nn.Module):
    """The rating-history CF expert used in the paper."""

    def __init__(self, user_count, item_count, dropout, dimension=64):
        super().__init__()
        self.gmf_user_embedding = nn.Embedding(user_count, dimension)
        self.gmf_item_embedding = nn.Embedding(item_count, dimension)
        self.mlp_user_embedding = nn.Embedding(user_count, dimension)
        self.mlp_item_embedding = nn.Embedding(item_count, dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, 32),
            nn.ReLU(),
        )
        self.prediction = nn.Linear(dimension + 32, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        embeddings = (
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        )
        for embedding in embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, user_ids, item_ids):
        gmf = self.gmf_user_embedding(user_ids) * self.gmf_item_embedding(item_ids)
        mlp_input = torch.cat(
            [self.mlp_user_embedding(user_ids), self.mlp_item_embedding(item_ids)], dim=1
        )
        mlp = self.mlp(mlp_input)
        return self.prediction(torch.cat([gmf, mlp], dim=1)).squeeze(1) + self.global_bias


class ReviewProjection(nn.Module):
    def __init__(self, input_dimension, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        return self.network(values)


class ReviewOnly(nn.Module):
    """Training-review user/item profiles followed by separate projections."""

    def __init__(self, user_profiles, item_profiles, dropout):
        super().__init__()
        dimension = int(user_profiles.shape[1])
        self.register_buffer("user_profiles", torch.as_tensor(user_profiles, dtype=torch.float32), persistent=False)
        self.register_buffer("item_profiles", torch.as_tensor(item_profiles, dtype=torch.float32), persistent=False)
        self.user_projection = ReviewProjection(dimension, dropout)
        self.item_projection = ReviewProjection(dimension, dropout)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, user_ids, item_ids):
        user_hidden = self.user_projection(self.user_profiles[user_ids])
        item_hidden = self.item_projection(self.item_profiles[item_ids])
        return self.prediction(torch.cat([user_hidden, item_hidden], dim=1)).squeeze(1)
