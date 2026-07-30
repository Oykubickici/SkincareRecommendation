from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

from .data import load_prepared, resolve_paths


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataset_overview(config: dict[str, Any], workspace: Path) -> Path:
    data = load_prepared(config, workspace)
    path = resolve_paths(config, workspace)["results"] / "figure_dataset_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    rating_counts = data.reviews["rating"].value_counts().sort_index()
    axes[0, 0].bar(rating_counts.index.astype(str), rating_counts.values)
    axes[0, 0].set_title("(a) Rating distribution")
    axes[0, 0].set_xlabel("Rating")
    axes[0, 0].set_ylabel("Reviews")

    activity = data.reviews.groupby("author_id").size()
    bins = np.logspace(0, np.log10(max(activity.max(), 2)), 35)
    axes[0, 1].hist(activity, bins=bins)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(b) User activity")
    axes[0, 1].set_xlabel("Reviews per user (log)")
    axes[0, 1].set_ylabel("Users (log)")

    popularity = data.reviews.groupby("product_id").size().sort_values(ascending=False)
    axes[1, 0].plot(np.arange(1, len(popularity) + 1), popularity.values)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(c) Product review long tail")
    axes[1, 0].set_xlabel("Product rank")
    axes[1, 0].set_ylabel("Reviews (log)")

    ingredient_counts: dict[str, int] = {}
    for tokens in data.products["ingredient_tokens"]:
        for token in tokens:
            ingredient_counts[token] = ingredient_counts.get(token, 0) + 1
    top = sorted(ingredient_counts.items(), key=lambda item: item[1], reverse=True)[:12]
    labels = [item[0] for item in top][::-1]
    values = [item[1] for item in top][::-1]
    axes[1, 1].barh(labels, values)
    axes[1, 1].set_title("(d) Frequent ingredient entities")
    axes[1, 1].set_xlabel("Products")
    axes[1, 1].tick_params(axis="y", labelsize=7)
    _save(fig, path)
    return path


def warm_start_figure(config: dict[str, Any], workspace: Path) -> Path | None:
    paths = resolve_paths(config, workspace)
    source = paths["results"] / "warm_start_metrics.csv"
    if not source.exists():
        return None
    metrics = pd.read_csv(source)
    path = paths["results"] / "figure_warm_start_metrics.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for name, group in metrics.groupby("model"):
        axes[0].plot(group["k"], group["precision"], marker="o", label=name)
        axes[1].plot(group["k"], group["ndcg"], marker="o", label=name)
    axes[0].set_title("(a) Precision@K")
    axes[1].set_title("(b) NDCG@K")
    for axis in axes:
        axis.set_xlabel("K")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(fontsize=6, loc="best")
    _save(fig, path)
    return path


def cold_start_figure(config: dict[str, Any], workspace: Path) -> Path | None:
    paths = resolve_paths(config, workspace)
    source = paths["results"] / "cold_start_metrics.csv"
    if not source.exists() or source.stat().st_size == 0:
        return None
    metrics = pd.read_csv(source)
    path = paths["results"] / "figure_cold_start_metrics.png"
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    subset = metrics[metrics["k"] == 10].sort_values("ndcg_mean")
    ax.barh(subset["model"], subset["ndcg_mean"], xerr=subset["ndcg_std"].fillna(0))
    ax.set_title("Item-held-out cold-start evaluation")
    ax.set_xlabel("NDCG@10 (mean across formulation-group folds)")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, path)
    return path


def reverse_figure(config: dict[str, Any], workspace: Path) -> Path | None:
    paths = resolve_paths(config, workspace)
    source = paths["results"] / "reverse_metrics.csv"
    if not source.exists() or source.stat().st_size == 0:
        return None
    metrics = pd.read_csv(source)
    path = paths["results"] / "figure_reverse_metrics.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for name, group in metrics.groupby("model"):
        axes[0].plot(group["m"], group["precision"], marker="o", label=name)
        axes[1].plot(group["m"], group["recall"], marker="o", label=name)
    axes[0].set_title("(a) Reverse precision")
    axes[1].set_title("(b) Reverse recall")
    for axis in axes:
        axis.set_xlabel("Target audience size M")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(fontsize=7)
    _save(fig, path)
    return path


def cluster_figures(config: dict[str, Any], workspace: Path) -> list[Path]:
    paths = resolve_paths(config, workspace)
    source = paths["results"] / "cluster_stability.csv"
    if not source.exists():
        return []
    metrics = pd.read_csv(source)
    summary = metrics.groupby("k", as_index=False).agg(
        silhouette=("silhouette", "mean"),
        silhouette_std=("silhouette", "std"),
        stability=("mean_pairwise_ari", "mean"),
    )
    stability_path = paths["results"] / "figure_cluster_stability.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].errorbar(
        summary["k"],
        summary["silhouette"],
        yerr=summary["silhouette_std"],
        marker="o",
    )
    axes[0].set_title("(a) Silhouette across seeds")
    axes[1].plot(summary["k"], summary["stability"], marker="o")
    axes[1].set_title("(b) Mean pairwise ARI")
    for axis in axes:
        axis.set_xlabel("Number of clusters")
        axis.grid(alpha=0.25)
    _save(fig, stability_path)

    data = load_prepared(config, workspace)
    profiles = normalize(data.train_matrix @ data.tfidf_features, axis=1).tocsr()
    best_k = int(summary.loc[summary["silhouette"].idxmax(), "k"])
    model = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=int(config["evaluation"]["primary_seed"]),
        n_init=5,
        batch_size=2048,
        max_iter=200,
    ).fit(profiles)
    centers = model.cluster_centers_
    top_indices = np.unique(
        np.argsort(centers, axis=1)[:, -4:].ravel()
    )
    display = centers[:, top_indices]
    heatmap_path = paths["results"] / "figure_cluster_heatmap.png"
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, best_k * 0.35)))
    image = ax.imshow(display, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(best_k))
    ax.set_yticklabels([f"Cluster {i}" for i in range(best_k)])
    ax.set_xticks(np.arange(len(top_indices)))
    ax.set_xticklabels(
        data.feature_names[top_indices], rotation=55, ha="right", fontsize=6
    )
    ax.set_title("Exploratory cluster-feature profiles")
    fig.colorbar(image, ax=ax, label="Cluster-center weight")
    _save(fig, heatmap_path)
    return [stability_path, heatmap_path]


def generate_all(config: dict[str, Any], workspace: Path) -> list[Path]:
    paths = [dataset_overview(config, workspace)]
    for result in [
        warm_start_figure(config, workspace),
        cold_start_figure(config, workspace),
        reverse_figure(config, workspace),
    ]:
        if result is not None:
            paths.append(result)
    paths.extend(cluster_figures(config, workspace))
    return paths
