from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


REVIEW_COLUMNS = [
    "author_id",
    "rating",
    "is_recommended",
    "submission_time",
    "review_text",
    "product_id",
    "product_name",
]


@dataclass
class PreparedData:
    """All objects required by the evaluation pipeline.

    The bundle contains only positive interactions for recommendation. Neutral
    and negative reviews are retained in the data audit, but never silently
    treated as positive preference evidence.
    """

    reviews: pd.DataFrame
    products: pd.DataFrame
    users: np.ndarray
    item_ids: np.ndarray
    train_user: np.ndarray
    train_item: np.ndarray
    val_user: np.ndarray
    val_item: np.ndarray
    test_user: np.ndarray
    test_item: np.ndarray
    train_matrix: sparse.csr_matrix
    binary_features: sparse.csr_matrix
    tfidf_features: sparse.csr_matrix
    bm25_features: sparse.csr_matrix
    feature_names: np.ndarray
    formulation_group: np.ndarray
    split_summary: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path.resolve())
    return config


def resolve_paths(config: dict[str, Any], workspace: str | Path) -> dict[str, Path]:
    workspace = Path(workspace).resolve()
    data_root = (workspace / config["data"]["root"]).resolve()
    return {
        "workspace": workspace,
        "data_root": data_root,
        "product_file": data_root / config["data"]["product_file"],
        "results": workspace / config["outputs"]["results_dir"],
        "reports": workspace / config["outputs"]["reports_dir"],
        "cache": workspace / config["outputs"]["cache_dir"],
    }


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def review_files(config: dict[str, Any], workspace: str | Path) -> list[Path]:
    paths = resolve_paths(config, workspace)
    return sorted(paths["data_root"].glob(config["data"]["review_glob"]))


def load_reviews(config: dict[str, Any], workspace: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in review_files(config, workspace):
        frame = pd.read_csv(
            path,
            usecols=REVIEW_COLUMNS,
            dtype={"author_id": "string", "product_id": "string"},
            low_memory=False,
        )
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No review CSV files matched the configured glob")
    reviews = pd.concat(frames, ignore_index=True)
    reviews["submission_time"] = pd.to_datetime(
        reviews["submission_time"], errors="coerce"
    )
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    return reviews


def load_products(config: dict[str, Any], workspace: str | Path) -> pd.DataFrame:
    path = resolve_paths(config, workspace)["product_file"]
    products = pd.read_csv(path, dtype={"product_id": "string"}, low_memory=False)
    if products["product_id"].duplicated().any():
        duplicates = int(products["product_id"].duplicated().sum())
        raise ValueError(f"product_info has {duplicates} duplicated product IDs")
    return products


def _split_outside_parentheses(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]" and depth:
            depth -= 1
        if char in ",;" and depth == 0:
            value = "".join(current).strip()
            if value:
                parts.append(value)
            current = []
        else:
            current.append(char)
    value = "".join(current).strip()
    if value:
        parts.append(value)
    return parts


_VARIATION_LABEL = re.compile(
    r"^(product\s+variation|variation|shade|size|ingredients?)\b.*:?$",
    flags=re.IGNORECASE,
)
_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_SPACE = re.compile(r"\s+")
_NON_TOKEN = re.compile(r"[^a-z0-9+\-/' ]+")


def normalize_ingredient(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = value.lower().replace("&", " and ")
    value = _PERCENT.sub(" ", value)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = _NON_TOKEN.sub(" ", value)
    value = _SPACE.sub(" ", value).strip(" -/'")
    value = re.sub(r"^(may contain|contains less than|active ingredient)\s*:?\s*", "", value)
    return value.strip()


def parse_ingredients(raw: Any) -> tuple[str, ...]:
    """Parse a product's declared INCI list into stable atomic entities.

    Ingredient order is deliberately not used by the model. Product variation
    labels are discarded, while ingredients from all declared variations are
    unioned. This avoids the legacy word-token/bigram representation.
    """

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    entries: list[str]
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            entries = [str(x) for x in parsed]
        else:
            entries = [str(parsed)]
    except (SyntaxError, ValueError):
        entries = [text]

    ingredients: set[str] = set()
    for entry in entries:
        stripped = entry.strip()
        if not stripped or _VARIATION_LABEL.match(stripped):
            continue
        if stripped.endswith(":") and len(_split_outside_parentheses(stripped)) == 1:
            continue
        for token in _split_outside_parentheses(stripped):
            normalized = normalize_ingredient(token)
            if (
                len(normalized) < 2
                or normalized in {"and", "or", "water aqua eau"}
                or normalized.startswith("product variation")
            ):
                if normalized == "water aqua eau":
                    ingredients.add("water")
                continue
            ingredients.add(normalized)
    return tuple(sorted(ingredients))


def build_audit(
    config: dict[str, Any], workspace: str | Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = resolve_paths(config, workspace)
    paths["results"].mkdir(parents=True, exist_ok=True)
    paths["reports"].mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(config, workspace)
    raw_counts = {
        "reviews": int(len(reviews)),
        "users": int(reviews["author_id"].nunique()),
        "products": int(reviews["product_id"].nunique()),
    }
    products = load_products(config, workspace)
    product_ids = set(products["product_id"].dropna())
    review_product_ids = set(reviews["product_id"].dropna())

    subset = ["author_id", "product_id", "submission_time", "review_text"]
    duplicate_reviews = int(reviews.duplicated(subset=subset, keep=False).sum())
    positive = reviews["rating"] >= config["data"]["positive_rating"]
    negative = reviews["rating"] <= config["data"]["negative_rating"]
    neutral = reviews["rating"] == 3

    flow_rows = [
        {
            "stage": "raw_review_files",
            "reviews": len(reviews),
            "users": reviews["author_id"].nunique(dropna=True),
            "products": reviews["product_id"].nunique(dropna=True),
            "removed_from_previous": 0,
        },
        {
            "stage": "valid_user_product_rating_date",
            "reviews": int(
                reviews[
                    reviews["author_id"].notna()
                    & reviews["product_id"].notna()
                    & reviews["rating"].notna()
                    & reviews["submission_time"].notna()
                ].shape[0]
            ),
            "users": reviews.loc[
                reviews["author_id"].notna()
                & reviews["product_id"].notna()
                & reviews["rating"].notna()
                & reviews["submission_time"].notna(),
                "author_id",
            ].nunique(),
            "products": reviews.loc[
                reviews["author_id"].notna()
                & reviews["product_id"].notna()
                & reviews["rating"].notna()
                & reviews["submission_time"].notna(),
                "product_id",
            ].nunique(),
        },
    ]
    flow_rows[1]["removed_from_previous"] = flow_rows[0]["reviews"] - flow_rows[1]["reviews"]

    audit = {
        "raw_reviews": int(len(reviews)),
        "raw_review_users": int(reviews["author_id"].nunique()),
        "raw_review_products": int(reviews["product_id"].nunique()),
        "metadata_products": int(products["product_id"].nunique()),
        "metadata_products_with_ingredients": int(products["ingredients"].notna().sum()),
        "review_products_missing_metadata": int(len(review_product_ids - product_ids)),
        "metadata_products_without_reviews": int(len(product_ids - review_product_ids)),
        "missing_author": int(reviews["author_id"].isna().sum()),
        "missing_product": int(reviews["product_id"].isna().sum()),
        "missing_rating": int(reviews["rating"].isna().sum()),
        "invalid_date": int(reviews["submission_time"].isna().sum()),
        "duplicate_review_rows_including_all_copies": duplicate_reviews,
        "positive_reviews_rating_ge_4": int(positive.sum()),
        "negative_reviews_rating_le_2": int(negative.sum()),
        "neutral_reviews_rating_eq_3": int(neutral.sum()),
        "date_min": reviews["submission_time"].min().date().isoformat(),
        "date_max": reviews["submission_time"].max().date().isoformat(),
        "source_sha256": {
            path.name: file_sha256(path)
            for path in [*review_files(config, workspace), paths["product_file"]]
        },
    }
    flow = pd.DataFrame(flow_rows)
    flow.to_csv(paths["results"] / "data_flow_initial.csv", index=False)
    with (paths["results"] / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    return flow, audit


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def labels(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.parent))]
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int32)


def formulation_groups(
    binary: sparse.csr_matrix, threshold: float
) -> np.ndarray:
    intersections = (binary @ binary.T).tocoo()
    lengths = np.asarray(binary.sum(axis=1)).ravel()
    union_find = _UnionFind(binary.shape[0])
    for row, col, intersection in zip(
        intersections.row, intersections.col, intersections.data
    ):
        if row >= col or intersection == 0:
            continue
        union = lengths[row] + lengths[col] - intersection
        if union and intersection / union >= threshold:
            union_find.union(int(row), int(col))
    return union_find.labels()


def _fit_feature_matrices(
    documents: list[tuple[str, ...]],
    training_items: np.ndarray,
    min_df: int,
    max_features: int | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    analyzer = lambda doc: doc  # noqa: E731 - required by sklearn API
    count_vectorizer = CountVectorizer(
        analyzer=analyzer, binary=True, lowercase=False, min_df=1
    )
    binary = count_vectorizer.fit_transform(documents).astype(np.float32).tocsr()

    tfidf_vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    train_documents = [documents[int(i)] for i in training_items]
    tfidf_vectorizer.fit(train_documents)
    tfidf = tfidf_vectorizer.transform(documents).astype(np.float32).tocsr()

    # BM25 item representation fitted only on training item document frequency.
    train_binary = binary[training_items]
    df = np.asarray((train_binary > 0).sum(axis=0)).ravel()
    n_docs = len(training_items)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    bm25 = binary.multiply(idf).tocsr()
    bm25_norm = np.sqrt(np.asarray(bm25.power(2).sum(axis=1)).ravel())
    bm25_norm[bm25_norm == 0] = 1
    bm25 = sparse.diags(1.0 / bm25_norm) @ bm25

    return (
        binary,
        tfidf,
        bm25.tocsr(),
        tfidf_vectorizer.get_feature_names_out(),
    )


def prepare_data(config: dict[str, Any], workspace: str | Path) -> PreparedData:
    paths = resolve_paths(config, workspace)
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(config, workspace)
    raw_counts = {
        "reviews": int(len(reviews)),
        "users": int(reviews["author_id"].nunique()),
        "products": int(reviews["product_id"].nunique()),
    }
    products_all = load_products(config, workspace)
    products_all["ingredient_tokens"] = products_all["ingredients"].map(parse_ingredients)
    products = products_all[
        products_all["product_id"].isin(reviews["product_id"].dropna().unique())
        & products_all["ingredient_tokens"].map(bool)
    ].copy()
    products = products.sort_values("product_id").reset_index(drop=True)

    reviews = reviews.merge(
        products[["product_id"]], on="product_id", how="inner", validate="many_to_one"
    )
    reviews = reviews.dropna(
        subset=["author_id", "product_id", "rating", "submission_time"]
    ).copy()
    reviews = reviews.drop_duplicates(
        subset=["author_id", "product_id", "submission_time", "review_text"],
        keep="first",
    )

    positive_threshold = config["data"]["positive_rating"]
    positives = reviews[reviews["rating"] >= positive_threshold].copy()
    positives = positives.sort_values(
        ["author_id", "submission_time", "product_id"], kind="mergesort"
    )
    # A user-product preference is one implicit interaction even when the
    # source contains multiple reviews at different dates. Keep the latest
    # positive evidence so the held-out target can never already be in train.
    positives = positives.drop_duplicates(
        subset=["author_id", "product_id"], keep="last"
    )
    unique_positives = positives.copy()
    positives["positive_count"] = positives.groupby("author_id")["product_id"].transform(
        "size"
    )
    positives = positives[
        positives["positive_count"]
        >= config["data"]["min_positive_interactions"]
    ].copy()
    positives["reverse_rank"] = positives.groupby("author_id").cumcount(
        ascending=False
    )

    user_ids = np.sort(positives["author_id"].unique())
    item_ids = products["product_id"].to_numpy(dtype=str)
    user_map = {value: idx for idx, value in enumerate(user_ids)}
    item_map = {value: idx for idx, value in enumerate(item_ids)}
    positives["user_idx"] = positives["author_id"].map(user_map).astype(np.int32)
    positives["item_idx"] = positives["product_id"].map(item_map).astype(np.int32)

    train = positives[positives["reverse_rank"] >= 2]
    validation = positives[positives["reverse_rank"] == 1]
    test = positives[positives["reverse_rank"] == 0]

    train_matrix = sparse.csr_matrix(
        (
            np.ones(len(train), dtype=np.float32),
            (train["user_idx"].to_numpy(), train["item_idx"].to_numpy()),
        ),
        shape=(len(user_ids), len(item_ids)),
    )
    train_matrix.data[:] = 1.0
    train_matrix.eliminate_zeros()

    training_items = np.unique(train["item_idx"].to_numpy())
    documents = products["ingredient_tokens"].tolist()
    selected_min_df = int(config["features"]["selected_min_df"])
    selected_cap = config["features"]["selected_feature_cap"]
    binary, tfidf, bm25, feature_names = _fit_feature_matrices(
        documents,
        training_items,
        min_df=selected_min_df,
        max_features=selected_cap,
    )
    groups = formulation_groups(
        binary, float(config["features"]["near_duplicate_jaccard"])
    )

    split_summary = {
        "eligible_users": int(len(user_ids)),
        "candidate_products": int(len(item_ids)),
        "train_interactions": int(len(train)),
        "validation_interactions": int(len(validation)),
        "test_interactions": int(len(test)),
        "train_products": int(train["item_idx"].nunique()),
        "validation_products": int(validation["item_idx"].nunique()),
        "test_products": int(test["item_idx"].nunique()),
        "tfidf_features": int(tfidf.shape[1]),
        "binary_ingredient_entities": int(binary.shape[1]),
        "exact_or_near_duplicate_groups": int(len(np.unique(groups))),
        "positive_definition": f"rating >= {positive_threshold}",
        "candidate_policy": "full ingredient-eligible review catalog minus seen train items",
    }
    with (paths["results"] / "split_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(split_summary, handle, indent=2)

    flow = pd.DataFrame(
        [
            {
                "stage": "raw_review_files",
                **raw_counts,
            },
            {
                "stage": "valid_reviews_with_ingredient_product",
                "reviews": len(reviews),
                "users": reviews["author_id"].nunique(),
                "products": reviews["product_id"].nunique(),
            },
            {
                "stage": "unique_positive_user_product_interactions",
                "reviews": len(unique_positives),
                "users": unique_positives["author_id"].nunique(),
                "products": unique_positives["product_id"].nunique(),
            },
            {
                "stage": "eligible_users_ge_3_positive",
                "reviews": len(positives),
                "users": positives["author_id"].nunique(),
                "products": positives["product_id"].nunique(),
            },
            {
                "stage": "train",
                "reviews": len(train),
                "users": train["author_id"].nunique(),
                "products": train["product_id"].nunique(),
            },
            {
                "stage": "validation",
                "reviews": len(validation),
                "users": validation["author_id"].nunique(),
                "products": validation["product_id"].nunique(),
            },
            {
                "stage": "test",
                "reviews": len(test),
                "users": test["author_id"].nunique(),
                "products": test["product_id"].nunique(),
            },
        ]
    )
    flow["removed_from_previous"] = np.nan
    for row_index in range(1, 4):
        flow.loc[row_index, "removed_from_previous"] = (
            flow.loc[row_index - 1, "reviews"] - flow.loc[row_index, "reviews"]
        )
    flow["partition_of"] = ""
    flow.loc[4:, "partition_of"] = "eligible_users_ge_3_positive"
    flow.to_csv(paths["results"] / "data_flow_final.csv", index=False)

    bundle = PreparedData(
        reviews=reviews,
        products=products,
        users=user_ids,
        item_ids=item_ids,
        train_user=train["user_idx"].to_numpy(dtype=np.int32),
        train_item=train["item_idx"].to_numpy(dtype=np.int32),
        val_user=validation["user_idx"].to_numpy(dtype=np.int32),
        val_item=validation["item_idx"].to_numpy(dtype=np.int32),
        test_user=test["user_idx"].to_numpy(dtype=np.int32),
        test_item=test["item_idx"].to_numpy(dtype=np.int32),
        train_matrix=train_matrix,
        binary_features=binary,
        tfidf_features=tfidf,
        bm25_features=bm25,
        feature_names=feature_names,
        formulation_group=groups,
        split_summary=split_summary,
    )
    pd.to_pickle(bundle, paths["cache"] / "prepared.pkl")
    products_export = products[
        ["product_id", "product_name", "brand_name", "primary_category", "secondary_category"]
    ].copy()
    products_export["ingredient_count"] = products["ingredient_tokens"].map(len)
    products_export["formulation_group"] = groups
    products_export.to_csv(paths["results"] / "product_catalog_audit.csv", index=False)
    return bundle


def load_prepared(config: dict[str, Any], workspace: str | Path) -> PreparedData:
    path = resolve_paths(config, workspace)["cache"] / "prepared.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; run the prepare command first"
        )
    return pd.read_pickle(path)


def rolling_cutoff_summary(
    reviews: pd.DataFrame, positive_rating: int, n_cutoffs: int
) -> pd.DataFrame:
    positives = reviews[
        (reviews["rating"] >= positive_rating) & reviews["submission_time"].notna()
    ].copy()
    quantiles = np.linspace(0.55, 0.75, n_cutoffs)
    cutoffs = positives["submission_time"].quantile(quantiles).drop_duplicates()
    rows: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        later = positives[positives["submission_time"] > cutoff]
        future_dates = later["submission_time"]
        validation_end = future_dates.quantile(0.5) if len(future_dates) else cutoff
        train = positives[positives["submission_time"] <= cutoff]
        validation = positives[
            (positives["submission_time"] > cutoff)
            & (positives["submission_time"] <= validation_end)
        ]
        test = positives[positives["submission_time"] > validation_end]
        rows.append(
            {
                "cutoff": pd.Timestamp(cutoff).date().isoformat(),
                "validation_end": pd.Timestamp(validation_end).date().isoformat(),
                "train_interactions": len(train),
                "validation_interactions": len(validation),
                "test_interactions": len(test),
                "train_users": train["author_id"].nunique(),
                "validation_users": validation["author_id"].nunique(),
                "test_users": test["author_id"].nunique(),
                "train_products": train["product_id"].nunique(),
                "validation_products": validation["product_id"].nunique(),
                "test_products": test["product_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)
