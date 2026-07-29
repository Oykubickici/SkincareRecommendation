from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from skincare_rec.data import (
    formulation_groups,
    parse_ingredients,
    prepare_data,
)


def test_ingredient_parser_uses_atomic_entities_not_word_bigrams():
    raw = "['Product variation 1:', 'Water (Aqua), Glycerin, Niacinamide']"
    tokens = parse_ingredients(raw)
    assert tokens == ("glycerin", "niacinamide", "water")
    assert "glycerin niacinamide" not in tokens


def test_formulation_groups_join_exact_and_near_duplicates():
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    groups = formulation_groups(matrix, threshold=0.95)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_prepare_is_chronological_and_fits_tfidf_on_training_products(tmp_path: Path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["one", "two", "three"],
            "brand_name": ["b", "b", "b"],
            "ingredients": [
                "['Water, Glycerin']",
                "['Niacinamide']",
                "['Retinol']",
            ],
            "primary_category": ["Skincare"] * 3,
            "secondary_category": ["Face"] * 3,
        }
    )
    products.to_csv(data_root / "product_info.csv", index=False)
    reviews = pd.DataFrame(
        {
            "author_id": ["u1", "u1", "u1"],
            "rating": [5, 5, 5],
            "is_recommended": [1, 1, 1],
            "submission_time": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "review_text": ["a", "b", "c"],
            "product_id": ["p1", "p2", "p3"],
            "product_name": ["one", "two", "three"],
        }
    )
    reviews.to_csv(data_root / "reviews_0-250.csv", index=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "data": {
            "root": "../raw",
            "product_file": "product_info.csv",
            "review_glob": "reviews_*.csv",
            "positive_rating": 4,
            "negative_rating": 2,
            "min_positive_interactions": 3,
        },
        "features": {
            "selected_min_df": 1,
            "selected_feature_cap": None,
            "near_duplicate_jaccard": 0.95,
        },
        "outputs": {
            "results_dir": "results",
            "reports_dir": "reports",
            "cache_dir": "data/cache",
        },
    }
    prepared = prepare_data(config, workspace)
    assert prepared.train_item.tolist() == [0]
    assert prepared.val_item.tolist() == [1]
    assert prepared.test_item.tolist() == [2]
    assert not np.asarray(
        prepared.train_matrix[prepared.test_user, prepared.test_item]
    ).ravel().any()
    assert set(prepared.feature_names) == {"water", "glycerin"}
    assert prepared.tfidf_features[1].nnz == 0
    assert prepared.tfidf_features[2].nnz == 0
