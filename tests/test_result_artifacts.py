from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_executed_warm_metrics_satisfy_leave_one_out_identities() -> None:
    metrics = pd.read_csv(ROOT / "results" / "warm_start_metrics.csv")
    assert set(metrics["model"]) == {
        "Random",
        "Popularity",
        "Binary Ingredient Cosine",
        "Ingredient Jaccard",
        "BM25 Ingredient",
        "TF-IDF Ingredient",
        "Item-kNN",
        "SVD",
        "Feature-aware BPR",
        "NCF",
        "LightGCN",
    }
    bounded = [
        "precision",
        "recall_hit_rate",
        "ndcg",
        "mrr",
        "catalog_coverage",
        "precision_ci_low",
        "precision_ci_high",
        "recall_hit_rate_ci_low",
        "recall_hit_rate_ci_high",
        "ndcg_ci_low",
        "ndcg_ci_high",
        "mrr_ci_low",
        "mrr_ci_high",
    ]
    assert np.isfinite(metrics[bounded].to_numpy()).all()
    assert ((metrics[bounded] >= 0) & (metrics[bounded] <= 1)).all().all()
    assert np.allclose(
        metrics["precision"] * metrics["k"],
        metrics["recall_hit_rate"],
        atol=1e-12,
    )
    assert (metrics["ndcg"] <= metrics["recall_hit_rate"] + 1e-12).all()
    for metric in ("precision", "recall_hit_rate", "ndcg", "mrr"):
        assert (metrics[f"{metric}_ci_low"] <= metrics[metric]).all()
        assert (metrics[metric] <= metrics[f"{metric}_ci_high"]).all()


def test_other_executed_metrics_are_bounded_and_splits_reconcile() -> None:
    cold = pd.read_csv(ROOT / "results" / "cold_start_metrics.csv")
    reverse = pd.read_csv(ROOT / "results" / "reverse_metrics.csv")
    for frame, columns in (
        (
            cold,
            [
                "precision_mean",
                "recall_mean",
                "ndcg_mean",
                "coverage_mean",
            ],
        ),
        (reverse, ["precision", "recall", "ndcg"]),
    ):
        assert np.isfinite(frame[columns].to_numpy()).all()
        assert ((frame[columns] >= 0) & (frame[columns] <= 1)).all().all()

    split = pd.read_json(
        ROOT / "results" / "split_summary.json", typ="series"
    )
    assert int(split["eligible_users"]) == int(split["validation_interactions"])
    assert int(split["eligible_users"]) == int(split["test_interactions"])
    assert int(split["train_interactions"]) > int(split["test_interactions"])
