from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize


class PopularityModel:
    name = "Popularity"

    def __init__(self, train: sparse.csr_matrix) -> None:
        self.popularity = np.asarray(train.sum(axis=0)).ravel().astype(np.float32)

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            self.popularity, (len(user_indices), len(self.popularity))
        ).copy()


class RandomModel:
    name = "Random"

    def __init__(self, n_items: int, seed: int) -> None:
        self.n_items = n_items
        self.seed = int(seed)

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        output = np.empty((len(user_indices), self.n_items), dtype=np.float32)
        for row, user in enumerate(user_indices):
            rng = np.random.default_rng(self.seed + int(user) * 1_000_003)
            output[row] = rng.random(self.n_items, dtype=np.float32)
        return output


class FeatureCosineModel:
    def __init__(
        self,
        name: str,
        train: sparse.csr_matrix,
        item_features: sparse.csr_matrix,
    ) -> None:
        self.name = name
        self.item_features = normalize(item_features, norm="l2", axis=1).tocsr()
        self.train = train

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        profiles = self.train[user_indices] @ self.item_features
        profiles = normalize(profiles, norm="l2", axis=1)
        return (profiles @ self.item_features.T).toarray()


class JaccardModel:
    name = "Ingredient Jaccard"

    def __init__(
        self, train: sparse.csr_matrix, binary_features: sparse.csr_matrix
    ) -> None:
        binary_features = (binary_features > 0).astype(np.float32).tocsr()
        self.train = train
        self.items = binary_features
        self.item_lengths = np.asarray(self.items.sum(axis=1)).ravel()

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        profiles = self.train[user_indices] @ self.items
        profiles.data[:] = 1
        profile_lengths = np.asarray(profiles.sum(axis=1)).ravel()
        intersection = (profiles @ self.items.T).toarray()
        union = (
            profile_lengths[:, None]
            + self.item_lengths[None, :]
            - intersection
        )
        return np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )


class ItemKNNModel:
    name = "Item-kNN"

    def __init__(self, train: sparse.csr_matrix, neighbors: int = 100) -> None:
        item_user = normalize(train.T, norm="l2", axis=1).tocsr()
        similarity = (item_user @ item_user.T).toarray().astype(np.float32)
        np.fill_diagonal(similarity, 0)
        neighbors = min(neighbors, similarity.shape[1] - 1)
        if neighbors > 0:
            cutoff = np.argpartition(
                similarity, kth=similarity.shape[1] - neighbors, axis=1
            )[:, :-neighbors]
            rows = np.arange(similarity.shape[0])[:, None]
            similarity[rows, cutoff] = 0
        self.similarity = sparse.csr_matrix(similarity)
        self.train = train

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return (self.train[user_indices] @ self.similarity).toarray()


class SVDModel:
    name = "SVD"

    def __init__(
        self, train: sparse.csr_matrix, factors: int = 64, seed: int = 2026
    ) -> None:
        factors = min(factors, min(train.shape) - 1)
        u, singular, vt = svds(
            train.astype(np.float64), k=factors, random_state=seed
        )
        order = np.argsort(singular)[::-1]
        self.user_factors = (u[:, order] * singular[order]).astype(np.float32)
        self.item_factors = vt[order].T.astype(np.float32)

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.user_factors[user_indices] @ self.item_factors.T


@dataclass
class FixedScoreModel:
    """Small deterministic model used by tests and audit utilities."""

    name: str
    scores: np.ndarray

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        return self.scores[user_indices].copy()
