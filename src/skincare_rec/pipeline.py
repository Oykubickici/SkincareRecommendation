from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, wilcoxon
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from .data import (
    PreparedData,
    _fit_feature_matrices,
    build_audit,
    load_prepared,
    prepare_data,
    resolve_paths,
    rolling_cutoff_summary,
)
from .metrics import (
    add_confidence_intervals,
    evaluate_leave_one_out,
)
from .models import (
    FeatureCosineModel,
    ItemKNNModel,
    JaccardModel,
    PopularityModel,
    RandomModel,
    SVDModel,
)
from .neural import FeatureAwareBPRModel, LightGCNModel, NCFModel


def run_audit(config: dict[str, Any], workspace: Path) -> None:
    _, audit = build_audit(config, workspace)
    paths = resolve_paths(config, workspace)
    prepared_reviews = None
    cache = paths["cache"] / "prepared.pkl"
    if cache.exists():
        prepared_reviews = load_prepared(config, workspace).reviews
    if prepared_reviews is not None:
        rolling = rolling_cutoff_summary(
            prepared_reviews,
            int(config["data"]["positive_rating"]),
            int(config["evaluation"]["temporal_cutoffs"]),
        )
        rolling.to_csv(paths["results"] / "temporal_cutoff_summary.csv", index=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def run_prepare(config: dict[str, Any], workspace: Path) -> PreparedData:
    bundle = prepare_data(config, workspace)
    rolling = rolling_cutoff_summary(
        bundle.reviews,
        int(config["data"]["positive_rating"]),
        int(config["evaluation"]["temporal_cutoffs"]),
    )
    rolling.to_csv(
        resolve_paths(config, workspace)["results"] / "temporal_cutoff_summary.csv",
        index=False,
    )
    print(json.dumps(bundle.split_summary, indent=2))
    return bundle


def _classical_models(
    data: PreparedData, config: dict[str, Any]
) -> list[Any]:
    seed = int(config["evaluation"]["primary_seed"])
    return [
        RandomModel(data.train_matrix.shape[1], seed),
        PopularityModel(data.train_matrix),
        FeatureCosineModel(
            "Binary Ingredient Cosine", data.train_matrix, data.binary_features
        ),
        JaccardModel(data.train_matrix, data.binary_features),
        FeatureCosineModel(
            "BM25 Ingredient", data.train_matrix, data.bm25_features
        ),
        FeatureCosineModel(
            "TF-IDF Ingredient", data.train_matrix, data.tfidf_features
        ),
        ItemKNNModel(
            data.train_matrix, int(config["models"]["item_knn_neighbors"])
        ),
        SVDModel(
            data.train_matrix,
            int(config["models"]["svd_factors"]),
            seed,
        ),
    ]


def _neural_model(
    name: str, data: PreparedData, config: dict[str, Any]
) -> Any:
    common = {
        "train": data.train_matrix,
        "dimension": int(config["models"]["neural_dimension"]),
        "epochs": int(config["models"]["neural_epochs"]),
        "learning_rate": float(config["models"]["neural_learning_rate"]),
        "seed": int(config["evaluation"]["primary_seed"]),
    }
    if name == "feature-bpr":
        return FeatureAwareBPRModel(
            item_features=data.tfidf_features,
            batch_size=int(config["models"]["neural_batch_size"]),
            **common,
        )
    if name == "ncf":
        return NCFModel(
            batch_size=int(config["models"]["neural_batch_size"]),
            **common,
        )
    if name == "lightgcn":
        return LightGCNModel(
            layers=int(config["models"]["lightgcn_layers"]),
            **common,
        )
    raise ValueError(f"Unknown neural model: {name}")


def _rank_biserial(differences: np.ndarray) -> float:
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def _holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _significance_table(per_model: dict[str, pd.DataFrame], k: int) -> pd.DataFrame:
    reference = "TF-IDF Ingredient"
    if reference not in per_model:
        return pd.DataFrame()
    reference_values = per_model[reference].set_index("user_idx")[f"ndcg@{k}"]
    rows: list[dict[str, Any]] = []
    for name, frame in per_model.items():
        if name == reference:
            continue
        values = frame.set_index("user_idx")[f"ndcg@{k}"]
        common = reference_values.index.intersection(values.index)
        diff = reference_values.loc[common].to_numpy() - values.loc[common].to_numpy()
        try:
            statistic, pvalue = wilcoxon(diff, zero_method="wilcox")
        except ValueError:
            statistic, pvalue = 0.0, 1.0
        rows.append(
            {
                "reference": reference,
                "comparison": name,
                "metric": f"NDCG@{k}",
                "n_users": len(common),
                "reference_mean": float(reference_values.loc[common].mean()),
                "comparison_mean": float(values.loc[common].mean()),
                "mean_difference": float(diff.mean()),
                "wilcoxon_statistic": float(statistic),
                "p_value": float(pvalue),
                "rank_biserial_effect": _rank_biserial(diff),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = _holm_adjust(result["p_value"].to_numpy())
    return result


def run_evaluation(
    config: dict[str, Any],
    workspace: Path,
    include_neural: list[str] | None = None,
) -> pd.DataFrame:
    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    paths["results"].mkdir(parents=True, exist_ok=True)
    models = _classical_models(data, config)
    include_neural = include_neural or []
    for name in include_neural:
        models.append(_neural_model(name, data, config))

    all_summaries: list[pd.DataFrame] = []
    per_model: dict[str, pd.DataFrame] = {}
    for model in models:
        print(f"Evaluating {model.name}...")
        result = evaluate_leave_one_out(
            model,
            data.train_matrix,
            data.test_user,
            data.test_item,
            [int(k) for k in config["evaluation"]["ks"]],
            int(config["evaluation"]["batch_size"]),
        )
        summary = add_confidence_intervals(
            result.summary,
            result.per_user,
            int(config["evaluation"]["bootstrap_samples"]),
            int(config["evaluation"]["primary_seed"]),
        )
        all_summaries.append(summary)
        safe = (
            model.name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        result.per_user.to_csv(
            paths["results"] / f"per_user_{safe}.csv", index=False
        )
        np.savez_compressed(
            paths["cache"] / f"recommendations_{safe}.npz",
            top_items=result.recommendation_items,
        )
        per_model[model.name] = result.per_user

    summary = pd.concat(all_summaries, ignore_index=True)
    summary.to_csv(paths["results"] / "warm_start_metrics.csv", index=False)
    significance = _significance_table(per_model, k=10)
    significance.to_csv(
        paths["results"] / "statistical_significance_ndcg10.csv", index=False
    )
    return summary


def run_feature_ablation(
    config: dict[str, Any],
    workspace: Path,
    sample_users: int = 10_000,
) -> pd.DataFrame:
    """Select TF-IDF vocabulary settings using validation data only.

    A fixed, seeded validation-user sample keeps the grid search tractable
    without consulting the test targets. The final test evaluation remains on
    every eligible user.
    """

    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    rng = np.random.default_rng(int(config["evaluation"]["primary_seed"]))
    count = min(sample_users, len(data.val_user))
    chosen = np.sort(rng.choice(len(data.val_user), size=count, replace=False))
    users = data.val_user[chosen]
    targets = data.val_item[chosen]
    documents = data.products["ingredient_tokens"].tolist()
    training_items = np.unique(data.train_item)
    rows: list[dict[str, Any]] = []
    for min_df in config["features"]["min_df_candidates"]:
        for cap in config["features"]["feature_cap_candidates"]:
            _, tfidf, _, names = _fit_feature_matrices(
                documents,
                training_items,
                int(min_df),
                None if cap is None else int(cap),
            )
            model = FeatureCosineModel(
                f"TF-IDF min_df={min_df} cap={cap}", data.train_matrix, tfidf
            )
            result = evaluate_leave_one_out(
                model,
                data.train_matrix,
                users,
                targets,
                [10],
                int(config["evaluation"]["batch_size"]),
            )
            row = result.summary.iloc[0].to_dict()
            row.update(
                {
                    "min_df": int(min_df),
                    "feature_cap": "none" if cap is None else int(cap),
                    "features": int(len(names)),
                    "selection_split": "validation",
                    "sample_users": int(count),
                    "seed": int(config["evaluation"]["primary_seed"]),
                }
            )
            rows.append(row)
            print(
                f"Ablation min_df={min_df}, cap={cap}, "
                f"NDCG@10={row['ndcg']:.6f}",
                flush=True,
            )
    result = pd.DataFrame(rows).sort_values(
        ["ndcg", "recall_hit_rate", "features"],
        ascending=[False, False, True],
    )
    result["selected"] = False
    if len(result):
        result.loc[result.index[0], "selected"] = True
    result.to_csv(paths["results"] / "tfidf_validation_ablation.csv", index=False)
    return result


def run_duplicate_sensitivity(
    config: dict[str, Any], workspace: Path
) -> pd.DataFrame:
    """Recompute warm metrics after removing formulation-overlap test cases."""

    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    train_groups: list[set[int]] = [set() for _ in range(len(data.users))]
    for user, item in zip(data.train_user, data.train_item):
        train_groups[int(user)].add(int(data.formulation_group[int(item)]))
    keep = np.array(
        [
            int(data.formulation_group[int(item)]) not in train_groups[int(user)]
            for user, item in zip(data.test_user, data.test_item)
        ],
        dtype=bool,
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(paths["cache"].glob("recommendations_*.npz")):
        top = np.load(path)["top_items"]
        if len(top) != len(keep):
            continue
        model_key = path.stem.removeprefix("recommendations_")
        for k in [int(value) for value in config["evaluation"]["ks"]]:
            recs = top[keep, :k]
            targets = data.test_item[keep]
            matches = recs == targets[:, None]
            hits = matches.any(axis=1)
            ranks = np.where(hits, matches.argmax(axis=1) + 1, 0)
            ndcg = np.zeros(len(ranks), dtype=float)
            mrr = np.zeros(len(ranks), dtype=float)
            ndcg[hits] = 1.0 / np.log2(ranks[hits] + 1)
            mrr[hits] = 1.0 / ranks[hits]
            rows.append(
                {
                    "model_key": model_key,
                    "k": k,
                    "users_retained": int(keep.sum()),
                    "users_removed_overlap": int((~keep).sum()),
                    "precision": float(hits.mean() / k),
                    "recall_hit_rate": float(hits.mean()),
                    "ndcg": float(ndcg.mean()),
                    "mrr": float(mrr.mean()),
                    "catalog_coverage": float(
                        np.unique(recs).size / len(data.item_ids)
                    ),
                    "test_formulation_absent_from_user_train_profile": True,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(
        paths["results"] / "duplicate_formulation_sensitivity.csv", index=False
    )
    return result


def run_temporal_evaluation(
    config: dict[str, Any], workspace: Path
) -> pd.DataFrame:
    """Evaluate temporal robustness at five global chronological cutoffs."""

    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    user_map = {value: idx for idx, value in enumerate(data.users)}
    item_map = {value: idx for idx, value in enumerate(data.item_ids)}
    positives = data.reviews[
        data.reviews["rating"] >= int(config["data"]["positive_rating"])
    ][["author_id", "product_id", "submission_time"]].copy()
    positives = positives[
        positives["author_id"].isin(user_map)
        & positives["product_id"].isin(item_map)
    ].sort_values(["author_id", "product_id", "submission_time"])
    positives = positives.drop_duplicates(
        ["author_id", "product_id"], keep="last"
    )
    positives["user_idx"] = positives["author_id"].map(user_map).astype(np.int32)
    positives["item_idx"] = positives["product_id"].map(item_map).astype(np.int32)
    quantiles = np.linspace(
        0.55, 0.75, int(config["evaluation"]["temporal_cutoffs"])
    )
    cutoffs = positives["submission_time"].quantile(quantiles).drop_duplicates()
    documents = data.products["ingredient_tokens"].tolist()
    rows: list[dict[str, Any]] = []
    for cutoff_index, cutoff in enumerate(cutoffs):
        future = positives[positives["submission_time"] > cutoff]
        validation_end = future["submission_time"].quantile(0.5)
        training = positives[positives["submission_time"] <= cutoff]
        test_pool = positives[positives["submission_time"] > validation_end]
        train_users = set(training["user_idx"].unique())
        test_pool = test_pool[test_pool["user_idx"].isin(train_users)]
        targets = (
            test_pool.sort_values(["submission_time", "product_id"])
            .groupby("user_idx", as_index=False)
            .first()
        )
        if not len(targets):
            continue
        train_matrix = sparse.csr_matrix(
            (
                np.ones(len(training), dtype=np.float32),
                (
                    training["user_idx"].to_numpy(),
                    training["item_idx"].to_numpy(),
                ),
            ),
            shape=data.train_matrix.shape,
        )
        train_matrix.data[:] = 1
        training_items = np.unique(training["item_idx"].to_numpy())
        _, tfidf, _, _ = _fit_feature_matrices(
            documents,
            training_items,
            int(config["features"]["selected_min_df"]),
            config["features"]["selected_feature_cap"],
        )
        models = [
            RandomModel(len(data.item_ids), int(config["evaluation"]["primary_seed"]) + cutoff_index),
            PopularityModel(train_matrix),
            FeatureCosineModel("TF-IDF Ingredient", train_matrix, tfidf),
            ItemKNNModel(
                train_matrix, int(config["models"]["item_knn_neighbors"])
            ),
        ]
        for model in models:
            result = evaluate_leave_one_out(
                model,
                train_matrix,
                targets["user_idx"].to_numpy(dtype=np.int32),
                targets["item_idx"].to_numpy(dtype=np.int32),
                [10],
                int(config["evaluation"]["batch_size"]),
            )
            row = result.summary.iloc[0].to_dict()
            row.update(
                {
                    "cutoff_index": cutoff_index + 1,
                    "cutoff": pd.Timestamp(cutoff).date().isoformat(),
                    "validation_end": pd.Timestamp(validation_end).date().isoformat(),
                    "train_interactions": int(len(training)),
                    "train_products": int(len(training_items)),
                    "candidate_policy": "full catalog minus user training items",
                }
            )
            rows.append(row)
            print(
                f"Temporal {cutoff_index + 1}/{len(cutoffs)} "
                f"{model.name}: NDCG@10={row['ndcg']:.6f}",
                flush=True,
            )
        del train_matrix, tfidf, models
        gc.collect()
    result = pd.DataFrame(rows)
    result.to_csv(paths["results"] / "temporal_metrics.csv", index=False)
    return result


def run_reverse_evaluation(
    config: dict[str, Any], workspace: Path, min_relevant_users: int = 5
) -> pd.DataFrame:
    """Leakage-audited product-to-user evaluation.

    For every target product, its contribution is subtracted from every
    candidate user's profile before similarity is calculated.
    """

    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    features = normalize(data.tfidf_features, axis=1).tocsr()
    profile_sums = (data.train_matrix @ features).tocsr()
    normalized_profiles = normalize(profile_sums, axis=1).tocsr()
    test_by_item: dict[int, np.ndarray] = {}
    for item in np.unique(data.test_item):
        relevant = data.test_user[data.test_item == item]
        if len(relevant) >= min_relevant_users:
            test_by_item[int(item)] = relevant

    ms = [int(value) for value in config["evaluation"]["reverse_ms"]]
    max_m = min(max(ms), data.train_matrix.shape[0])
    rng = np.random.default_rng(int(config["evaluation"]["primary_seed"]))
    activity = np.asarray(data.train_matrix.sum(axis=1)).ravel()
    rows: list[dict[str, Any]] = []
    for target_item, relevant_users in test_by_item.items():
        target = features[target_item]
        content_scores = (normalized_profiles @ target.T).toarray().ravel()
        users_with_target = data.train_matrix[:, target_item].nonzero()[0]
        for user in users_with_target:
            adjusted_row = profile_sums.getrow(int(user)) - target
            adjusted_row = normalize(adjusted_row, axis=1)
            content_scores[int(user)] = float((adjusted_row @ target.T).toarray()[0, 0])
        model_scores = {
            "TF-IDF Reverse": content_scores,
            "User Activity": activity,
            "Random Reverse": rng.random(len(activity)),
        }
        relevant_set = set(int(x) for x in relevant_users)
        for model_name, scores in model_scores.items():
            top = np.argpartition(-scores, max_m - 1)[:max_m]
            top = top[np.argsort(-scores[top])]
            for m in ms:
                selected = top[: min(m, len(top))]
                hits = np.array([int(x) in relevant_set for x in selected])
                hit_count = int(hits.sum())
                ranks = np.flatnonzero(hits) + 1
                dcg = float((1.0 / np.log2(ranks + 1)).sum()) if len(ranks) else 0.0
                ideal_count = min(len(relevant_set), m)
                idcg = float(
                    (1.0 / np.log2(np.arange(ideal_count) + 2)).sum()
                )
                rows.append(
                    {
                        "model": model_name,
                        "target_item_idx": target_item,
                        "m": m,
                        "relevant_users": len(relevant_set),
                        "hits": hit_count,
                        "precision": hit_count / m,
                        "recall": hit_count / len(relevant_set),
                        "ndcg": dcg / idcg if idcg else 0.0,
                        "target_removed_from_profiles": True,
                        "candidate_users": len(activity),
                    }
                )
    per_product = pd.DataFrame(rows)
    per_product.to_csv(paths["results"] / "reverse_per_product.csv", index=False)
    summary = (
        per_product.groupby(["model", "m"], as_index=False)
        .agg(
            products=("target_item_idx", "nunique"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            ndcg=("ndcg", "mean"),
            candidate_users=("candidate_users", "first"),
        )
        if len(per_product)
        else pd.DataFrame()
    )
    summary.to_csv(paths["results"] / "reverse_metrics.csv", index=False)
    return summary


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    selected = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    selected_scores = np.take_along_axis(scores, selected, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(selected, order, axis=1)


def run_cold_start_evaluation(
    config: dict[str, Any], workspace: Path, folds: int = 5
) -> pd.DataFrame:
    """Evaluate genuinely unseen formulation groups.

    All positive interactions for a held-out formulation group are removed
    before profiles and IDF statistics are created. Ranking is performed only
    among the fold's unseen products, which is stated explicitly in the output.
    """

    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    item_map = {item: idx for idx, item in enumerate(data.item_ids)}
    user_map = {user: idx for idx, user in enumerate(data.users)}
    positives = data.reviews[
        data.reviews["rating"] >= int(config["data"]["positive_rating"])
    ][["author_id", "product_id"]].drop_duplicates()
    positives = positives[
        positives["author_id"].isin(user_map)
        & positives["product_id"].isin(item_map)
    ].copy()
    positives["user_idx"] = positives["author_id"].map(user_map).astype(np.int32)
    positives["item_idx"] = positives["product_id"].map(item_map).astype(np.int32)
    documents = data.products["ingredient_tokens"].tolist()
    max_k = max(int(k) for k in config["evaluation"]["ks"])
    rows: list[dict[str, Any]] = []
    seed = int(config["evaluation"]["primary_seed"])

    for fold in range(folds):
        print(f"Cold-start fold {fold + 1}/{folds}", flush=True)
        cold_mask = (data.formulation_group % folds) == fold
        cold_items = np.flatnonzero(cold_mask).astype(np.int32)
        warm_items = np.flatnonzero(~cold_mask).astype(np.int32)
        fold_interactions = positives[
            positives["item_idx"].isin(cold_items)
        ]
        training = positives[~positives["item_idx"].isin(cold_items)]
        users_with_train = set(training["user_idx"].unique())
        users_with_test = set(fold_interactions["user_idx"].unique())
        eligible_users = np.array(
            sorted(users_with_train & users_with_test), dtype=np.int32
        )
        if not len(eligible_users) or not len(cold_items):
            continue
        training = training[training["user_idx"].isin(eligible_users)]
        fold_interactions = fold_interactions[
            fold_interactions["user_idx"].isin(eligible_users)
        ]
        train_matrix = sparse.csr_matrix(
            (
                np.ones(len(training), dtype=np.float32),
                (
                    training["user_idx"].to_numpy(),
                    training["item_idx"].to_numpy(),
                ),
            ),
            shape=data.train_matrix.shape,
        )
        train_matrix.data[:] = 1
        binary, tfidf, bm25, _ = _fit_feature_matrices(
            documents,
            warm_items,
            int(config["features"]["selected_min_df"]),
            config["features"]["selected_feature_cap"],
        )
        model_factories = [
            lambda: RandomModel(len(data.item_ids), seed + fold),
            lambda: PopularityModel(train_matrix),
            lambda: FeatureCosineModel(
                "Binary Ingredient Cosine", train_matrix, binary
            ),
            lambda: JaccardModel(train_matrix, binary),
            lambda: FeatureCosineModel(
                "BM25 Ingredient", train_matrix, bm25
            ),
            lambda: FeatureCosineModel(
                "TF-IDF Ingredient", train_matrix, tfidf
            ),
        ]
        relevant = (
            fold_interactions.groupby("user_idx")["item_idx"]
            .apply(lambda values: set(int(x) for x in values))
            .to_dict()
        )
        batch_size = int(config["evaluation"]["batch_size"])
        for build_model in model_factories:
            model = build_model()
            print(f"  evaluating {model.name}", flush=True)
            all_recs: list[np.ndarray] = []
            for start in range(0, len(eligible_users), batch_size):
                users = eligible_users[start : start + batch_size]
                scores = model.score(users)[:, cold_items]
                local_top = _topk(scores, max_k)
                all_recs.append(cold_items[local_top])
            recommendations = np.vstack(all_recs)
            for k in [int(value) for value in config["evaluation"]["ks"]]:
                precisions = []
                recalls = []
                ndcgs = []
                for user, recs in zip(eligible_users, recommendations[:, :k]):
                    relevant_set = relevant[int(user)]
                    hits = np.array([int(item) in relevant_set for item in recs])
                    hit_count = int(hits.sum())
                    precisions.append(hit_count / k)
                    recalls.append(hit_count / len(relevant_set))
                    hit_ranks = np.flatnonzero(hits) + 1
                    dcg = (
                        float((1 / np.log2(hit_ranks + 1)).sum())
                        if len(hit_ranks)
                        else 0.0
                    )
                    ideal = min(len(relevant_set), k)
                    idcg = float((1 / np.log2(np.arange(ideal) + 2)).sum())
                    ndcgs.append(dcg / idcg if idcg else 0.0)
                rows.append(
                    {
                        "fold": fold,
                        "model": model.name,
                        "k": k,
                        "users": len(eligible_users),
                        "cold_products": len(cold_items),
                        "train_products": len(warm_items),
                        "precision": float(np.mean(precisions)),
                        "recall": float(np.mean(recalls)),
                        "ndcg": float(np.mean(ndcgs)),
                        "catalog_coverage": float(
                            np.unique(recommendations[:, :k]).size
                            / len(cold_items)
                        ),
                        "all_cold_interactions_removed_from_training": True,
                        "formulation_group_isolated": True,
                        "candidate_policy": "all unseen products in held-out fold",
                    }
                )
            del model
        del binary, tfidf, bm25, train_matrix, model_factories
        gc.collect()
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(paths["results"] / "cold_start_metrics_per_fold.csv", index=False)
    summary = (
        per_fold.groupby(["model", "k"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            users_mean=("users", "mean"),
            cold_products_mean=("cold_products", "mean"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            ndcg_mean=("ndcg", "mean"),
            ndcg_std=("ndcg", "std"),
            coverage_mean=("catalog_coverage", "mean"),
        )
        if len(per_fold)
        else pd.DataFrame()
    )
    summary.to_csv(paths["results"] / "cold_start_metrics.csv", index=False)
    return summary


def run_cluster_stability(
    config: dict[str, Any], workspace: Path
) -> pd.DataFrame:
    data = load_prepared(config, workspace)
    paths = resolve_paths(config, workspace)
    profiles = normalize(data.train_matrix @ data.tfidf_features, axis=1).tocsr()
    seed = int(config["evaluation"]["primary_seed"])
    rng = np.random.default_rng(seed)
    sample_size = min(10_000, profiles.shape[0])
    sample_indices = np.sort(
        rng.choice(profiles.shape[0], size=sample_size, replace=False)
    )
    sample = profiles[sample_indices]
    dense_validation = sample[
        np.sort(rng.choice(sample_size, size=min(2_000, sample_size), replace=False))
    ].toarray()
    seeds = [int(x) for x in config["evaluation"]["seeds"]]
    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, list[np.ndarray]] = {}
    for k in range(2, 16):
        labels_by_k[k] = []
        for model_seed in seeds:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=model_seed,
                n_init=5,
                batch_size=2048,
                max_iter=200,
            )
            labels = model.fit_predict(sample)
            labels_by_k[k].append(labels)
            validation_labels = model.predict(dense_validation)
            rows.append(
                {
                    "k": k,
                    "seed": model_seed,
                    "inertia": float(model.inertia_),
                    "silhouette": float(
                        silhouette_score(
                            dense_validation,
                            validation_labels,
                            metric="cosine",
                        )
                    ),
                    "davies_bouldin": float(
                        davies_bouldin_score(dense_validation, validation_labels)
                    ),
                    "calinski_harabasz": float(
                        calinski_harabasz_score(
                            dense_validation, validation_labels
                        )
                    ),
                }
            )
        pairwise = []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                pairwise.append(
                    adjusted_rand_score(
                        labels_by_k[k][left], labels_by_k[k][right]
                    )
                )
        for row in rows[-len(seeds) :]:
            row["mean_pairwise_ari"] = float(np.mean(pairwise))
    result = pd.DataFrame(rows)
    result.to_csv(paths["results"] / "cluster_stability.csv", index=False)

    # Bootstrap stability for the originally claimed nine-cluster solution.
    bootstrap_labels: list[np.ndarray] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for bootstrap_index, model_seed in enumerate(seeds):
        bootstrap_rng = np.random.default_rng(model_seed)
        indices = bootstrap_rng.integers(0, sample_size, size=sample_size)
        model = MiniBatchKMeans(
            n_clusters=9,
            random_state=model_seed,
            n_init=5,
            batch_size=2048,
            max_iter=200,
        )
        model.fit(sample[indices])
        labels = model.predict(dense_validation)
        bootstrap_labels.append(labels)
        bootstrap_rows.append(
            {
                "bootstrap": bootstrap_index + 1,
                "seed": model_seed,
                "sample_size": sample_size,
                "validation_users": len(dense_validation),
                "silhouette": float(
                    silhouette_score(dense_validation, labels, metric="cosine")
                ),
            }
        )
    bootstrap_aris = []
    for left in range(len(bootstrap_labels)):
        for right in range(left + 1, len(bootstrap_labels)):
            bootstrap_aris.append(
                adjusted_rand_score(
                    bootstrap_labels[left], bootstrap_labels[right]
                )
            )
    for row in bootstrap_rows:
        row["mean_pairwise_bootstrap_ari"] = float(np.mean(bootstrap_aris))
    pd.DataFrame(bootstrap_rows).to_csv(
        paths["results"] / "cluster_bootstrap_stability.csv", index=False
    )
    return result
