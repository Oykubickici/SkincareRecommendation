from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse


class BatchScorer(Protocol):
    name: str

    def score(self, user_indices: np.ndarray) -> np.ndarray:
        """Return dense scores with shape (len(user_indices), n_items)."""


@dataclass
class EvaluationResult:
    summary: pd.DataFrame
    per_user: pd.DataFrame
    recommendation_items: np.ndarray


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)[:, :k]
    partition = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(scores, partition, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(partition, order, axis=1)


def evaluate_leave_one_out(
    model: BatchScorer,
    train_matrix: sparse.csr_matrix,
    test_user: np.ndarray,
    test_item: np.ndarray,
    ks: list[int],
    batch_size: int,
) -> EvaluationResult:
    """Evaluate one relevant test item per user against the full catalog."""

    if len(test_user) != len(test_item):
        raise ValueError("test_user and test_item lengths differ")
    if len(np.unique(test_user)) != len(test_user):
        raise ValueError("leave-one-out evaluation requires one row per user")

    max_k = max(ks)
    all_top = np.empty((len(test_user), max_k), dtype=np.int32)
    cursor = 0
    for start in range(0, len(test_user), batch_size):
        stop = min(start + batch_size, len(test_user))
        users = test_user[start:stop]
        scores = np.asarray(model.score(users), dtype=np.float32)
        if scores.shape != (len(users), train_matrix.shape[1]):
            raise ValueError(
                f"{model.name} returned {scores.shape}, expected "
                f"{(len(users), train_matrix.shape[1])}"
            )
        train_slice = train_matrix[users]
        rows, cols = train_slice.nonzero()
        scores[rows, cols] = -np.inf
        if not np.isfinite(scores).any(axis=1).all():
            raise ValueError(f"{model.name} produced a user with no finite candidates")
        top = _topk_indices(scores, max_k)
        all_top[cursor : cursor + len(users)] = top
        cursor += len(users)

    per_user = pd.DataFrame({"user_idx": test_user, "test_item": test_item})
    summaries: list[dict[str, float | int | str]] = []
    for k in ks:
        recs = all_top[:, :k]
        matches = recs == test_item[:, None]
        hit = matches.any(axis=1).astype(np.float64)
        rank = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)
        ndcg = np.zeros(len(rank), dtype=np.float64)
        mrr = np.zeros(len(rank), dtype=np.float64)
        retrieved = rank > 0
        ndcg[retrieved] = 1.0 / np.log2(rank[retrieved] + 1)
        mrr[retrieved] = 1.0 / rank[retrieved]
        precision = hit / k
        recall = hit.copy()
        per_user[f"hit@{k}"] = hit
        per_user[f"precision@{k}"] = precision
        per_user[f"recall@{k}"] = recall
        per_user[f"ndcg@{k}"] = ndcg
        per_user[f"mrr@{k}"] = mrr
        coverage = np.unique(recs).size / train_matrix.shape[1]
        summaries.append(
            {
                "model": model.name,
                "k": k,
                "users": len(test_user),
                "precision": float(precision.mean()),
                "recall_hit_rate": float(recall.mean()),
                "ndcg": float(ndcg.mean()),
                "mrr": float(mrr.mean()),
                "catalog_coverage": float(coverage),
            }
        )

    summary = pd.DataFrame(summaries)
    validate_leave_one_out_metrics(summary)
    return EvaluationResult(summary, per_user, all_top)


def validate_leave_one_out_metrics(summary: pd.DataFrame, atol: float = 1e-10) -> None:
    metric_columns = [
        "precision",
        "recall_hit_rate",
        "ndcg",
        "mrr",
        "catalog_coverage",
    ]
    values = summary[metric_columns].to_numpy(float)
    if not ((values >= -atol) & (values <= 1 + atol)).all():
        raise AssertionError("ranking metrics must be in [0, 1]")
    expected_precision = summary["recall_hit_rate"] / summary["k"]
    if not np.allclose(summary["precision"], expected_precision, atol=atol):
        raise AssertionError("Precision@K must equal HitRate@K / K")
    if (summary["ndcg"] > summary["recall_hit_rate"] + atol).any():
        raise AssertionError("NDCG@K cannot exceed HitRate@K in leave-one-out")


def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    chunk = min(200, samples)
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def add_confidence_intervals(
    summary: pd.DataFrame,
    per_user: pd.DataFrame,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    output = summary.copy()
    for idx, row in output.iterrows():
        k = int(row["k"])
        mapping = {
            "precision": f"precision@{k}",
            "recall_hit_rate": f"recall@{k}",
            "ndcg": f"ndcg@{k}",
            "mrr": f"mrr@{k}",
        }
        for metric, column in mapping.items():
            low, high = bootstrap_mean_ci(
                per_user[column].to_numpy(),
                samples=samples,
                seed=seed + idx,
            )
            output.loc[idx, f"{metric}_ci_low"] = low
            output.loc[idx, f"{metric}_ci_high"] = high
    return output
