"""Dependency-free local test runner for environments without pytest."""

from pathlib import Path
from tempfile import TemporaryDirectory

from test_data import (
    test_formulation_groups_join_exact_and_near_duplicates,
    test_ingredient_parser_uses_atomic_entities_not_word_bigrams,
    test_prepare_is_chronological_and_fits_tfidf_on_training_products,
)
from test_metrics import (
    test_duplicate_test_users_are_rejected,
    test_leave_one_out_metric_identities_and_seen_masking,
)
from test_manuscript_guardrails import (
    test_revised_manuscript_has_no_known_unsupported_claims,
    test_revised_manuscript_references_resolve,
)
from test_result_artifacts import (
    test_executed_warm_metrics_satisfy_leave_one_out_identities,
    test_other_executed_metrics_are_bounded_and_splits_reconcile,
)


def main() -> None:
    tests = [
        test_ingredient_parser_uses_atomic_entities_not_word_bigrams,
        test_formulation_groups_join_exact_and_near_duplicates,
        test_leave_one_out_metric_identities_and_seen_masking,
        test_duplicate_test_users_are_rejected,
        test_revised_manuscript_has_no_known_unsupported_claims,
        test_revised_manuscript_references_resolve,
        test_executed_warm_metrics_satisfy_leave_one_out_identities,
        test_other_executed_metrics_are_bounded_and_splits_reconcile,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    with TemporaryDirectory() as directory:
        test_prepare_is_chronological_and_fits_tfidf_on_training_products(
            Path(directory)
        )
    print(
        "PASS test_prepare_is_chronological_and_fits_tfidf_on_training_products"
    )


if __name__ == "__main__":
    main()
