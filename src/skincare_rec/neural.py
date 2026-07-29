from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import torch
from torch import nn


def _negative_items(
    train: sparse.csr_matrix,
    users: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, train.shape[1], size=len(users), dtype=np.int64)
    for idx, user in enumerate(users):
        start, stop = train.indptr[user], train.indptr[user + 1]
        seen = train.indices[start:stop]
        while np.any(seen == negatives[idx]):
            negatives[idx] = rng.integers(0, train.shape[1])
    return negatives


class _BPRBase:
    name: str

    def __init__(self, train: sparse.csr_matrix, device: str = "cpu") -> None:
        self.train = train
        self.device = torch.device(device)

    @staticmethod
    def _bpr_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return -torch.nn.functional.logsigmoid(positive - negative).mean()


class HybridLightFMModel(_BPRBase):
    """Feature-aware latent factor model trained with BPR.

    This is the reproducible hybrid baseline used in place of an unavailable
    platform-specific LightFM binary. Item representations combine a free item
    embedding and a linear projection of training-fitted ingredient features.
    """

    name = "Hybrid LightFM"

    def __init__(
        self,
        train: sparse.csr_matrix,
        item_features: sparse.csr_matrix,
        dimension: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        super().__init__(train)
        torch.manual_seed(seed)
        self.user_embedding = nn.Embedding(train.shape[0], dimension)
        self.item_embedding = nn.Embedding(train.shape[1], dimension)
        self.feature_projection = nn.Linear(
            item_features.shape[1], dimension, bias=False
        )
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.normal_(self.feature_projection.weight, std=0.01)
        self.features = torch.tensor(
            item_features.toarray(), dtype=torch.float32
        )
        parameters = [
            *self.user_embedding.parameters(),
            *self.item_embedding.parameters(),
            *self.feature_projection.parameters(),
        ]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        rng = np.random.default_rng(seed)
        rows, cols = train.nonzero()
        for _ in range(epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                users = rows[batch].astype(np.int64)
                positive = cols[batch].astype(np.int64)
                negative = _negative_items(train, users, rng)
                u = torch.from_numpy(users)
                p = torch.from_numpy(positive)
                n = torch.from_numpy(negative)
                user_vec = self.user_embedding(u)
                pos_vec = self.item_embedding(p) + self.feature_projection(
                    self.features[p]
                )
                neg_vec = self.item_embedding(n) + self.feature_projection(
                    self.features[n]
                )
                pos_score = (user_vec * pos_vec).sum(dim=1)
                neg_score = (user_vec * neg_vec).sum(dim=1)
                loss = self._bpr_loss(pos_score, neg_score)
                loss = loss + 1e-6 * (
                    user_vec.square().mean()
                    + pos_vec.square().mean()
                    + neg_vec.square().mean()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.user_factors = self.user_embedding.weight.detach().cpu().numpy()
        item_tensor = torch.arange(train.shape[1])
        self.item_factors = (
            self.item_embedding(item_tensor)
            + self.feature_projection(self.features)
        ).detach().cpu().numpy()

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors[user_indices] @ self.item_factors.T


class _NCFNetwork(nn.Module):
    def __init__(self, n_users: int, n_items: int, dimension: int) -> None:
        super().__init__()
        self.user = nn.Embedding(n_users, dimension)
        self.item = nn.Embedding(n_items, dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension * 2, dimension * 2),
            nn.ReLU(),
            nn.Linear(dimension * 2, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1),
        )
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([self.user(users), self.item(items)], dim=-1)
        return self.mlp(joined).squeeze(-1)


class NCFModel(_BPRBase):
    name = "NCF"

    def __init__(
        self,
        train: sparse.csr_matrix,
        dimension: int,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> None:
        super().__init__(train)
        torch.manual_seed(seed)
        self.network = _NCFNetwork(
            train.shape[0], train.shape[1], dimension
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)
        rows, cols = train.nonzero()
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            order = rng.permutation(len(rows))
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                users = rows[batch].astype(np.int64)
                positive = cols[batch].astype(np.int64)
                negative = _negative_items(train, users, rng)
                all_users = torch.from_numpy(np.concatenate([users, users]))
                all_items = torch.from_numpy(np.concatenate([positive, negative]))
                labels = torch.cat(
                    [torch.ones(len(users)), torch.zeros(len(users))]
                )
                predictions = self.network(all_users, all_items)
                loss = loss_fn(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.network.eval()

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        n_items = self.train.shape[1]
        output = np.empty((len(user_indices), n_items), dtype=np.float32)
        item_indices = torch.arange(n_items, dtype=torch.long)
        with torch.no_grad():
            for row, user in enumerate(user_indices):
                users = torch.full((n_items,), int(user), dtype=torch.long)
                output[row] = self.network(users, item_indices).numpy()
        return output


class LightGCNModel(_BPRBase):
    name = "LightGCN"

    def __init__(
        self,
        train: sparse.csr_matrix,
        dimension: int,
        layers: int,
        epochs: int,
        learning_rate: float,
        seed: int,
        max_training_edges: int = 250_000,
    ) -> None:
        super().__init__(train)
        torch.manual_seed(seed)
        self.n_users, self.n_items = train.shape
        self.layers = layers
        self.embedding = nn.Embedding(self.n_users + self.n_items, dimension)
        nn.init.normal_(self.embedding.weight, std=0.05)
        self.adjacency = self._normalized_adjacency(train)
        optimizer = torch.optim.Adam(self.embedding.parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)
        rows, cols = train.nonzero()
        for _ in range(epochs):
            if len(rows) > max_training_edges:
                chosen = rng.choice(
                    len(rows), size=max_training_edges, replace=False
                )
                users = rows[chosen].astype(np.int64)
                positive = cols[chosen].astype(np.int64)
            else:
                users = rows.astype(np.int64)
                positive = cols.astype(np.int64)
            negative = _negative_items(train, users, rng)
            propagated = self._propagate()
            user_vec = propagated[torch.from_numpy(users)]
            pos_vec = propagated[
                torch.from_numpy(positive + self.n_users)
            ]
            neg_vec = propagated[
                torch.from_numpy(negative + self.n_users)
            ]
            loss = self._bpr_loss(
                (user_vec * pos_vec).sum(dim=1),
                (user_vec * neg_vec).sum(dim=1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            propagated = self._propagate().cpu().numpy()
        self.user_factors = propagated[: self.n_users]
        self.item_factors = propagated[self.n_users :]

    def _normalized_adjacency(self, train: sparse.csr_matrix) -> torch.Tensor:
        rows, cols = train.nonzero()
        left = np.concatenate([rows, cols + self.n_users])
        right = np.concatenate([cols + self.n_users, rows])
        degree = np.bincount(
            left, minlength=self.n_users + self.n_items
        ).astype(np.float32)
        degree[degree == 0] = 1
        values = 1.0 / np.sqrt(degree[left] * degree[right])
        indices = torch.tensor(np.vstack([left, right]), dtype=torch.long)
        values_tensor = torch.tensor(values, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices,
            values_tensor,
            (self.n_users + self.n_items, self.n_users + self.n_items),
        ).coalesce()

    def _propagate(self) -> torch.Tensor:
        embeddings = [self.embedding.weight]
        current = self.embedding.weight
        for _ in range(self.layers):
            current = torch.sparse.mm(self.adjacency, current)
            embeddings.append(current)
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors[user_indices] @ self.item_factors.T
