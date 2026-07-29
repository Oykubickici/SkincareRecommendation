import numpy as np
from scipy import sparse

from skincare_rec.metrics import evaluate_leave_one_out
from skincare_rec.models import FixedScoreModel


def test_leave_one_out_metric_identities_and_seen_masking():
    train = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    scores = np.array(
        [
            [100, 0.8, 0.9, 0.7],
            [0.8, 100, 0.7, 0.9],
        ],
        dtype=np.float32,
    )
    model = FixedScoreModel("toy", scores)
    result = evaluate_leave_one_out(
        model,
        train,
        test_user=np.array([0, 1]),
        test_item=np.array([2, 3]),
        ks=[1, 2],
        batch_size=2,
    )
    at_one = result.summary[result.summary["k"] == 1].iloc[0]
    assert at_one["precision"] == 1.0
    assert at_one["recall_hit_rate"] == 1.0
    assert at_one["ndcg"] == 1.0
    assert np.allclose(
        result.summary["precision"],
        result.summary["recall_hit_rate"] / result.summary["k"],
    )


def test_duplicate_test_users_are_rejected():
    train = sparse.csr_matrix(np.zeros((1, 2), dtype=np.float32))
    model = FixedScoreModel("toy", np.zeros((1, 2), dtype=np.float32))
    try:
        evaluate_leave_one_out(
            model,
            train,
            test_user=np.array([0, 0]),
            test_item=np.array([0, 1]),
            ks=[1],
            batch_size=2,
        )
    except ValueError as error:
        assert "one row per user" in str(error)
    else:
        raise AssertionError("duplicate users must be rejected")
