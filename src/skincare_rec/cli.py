from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_config
from .figures import generate_all
from .pipeline import (
    run_audit,
    run_cluster_stability,
    run_cold_start_evaluation,
    run_duplicate_sensitivity,
    run_evaluation,
    run_feature_ablation,
    run_prepare,
    run_reverse_evaluation,
    run_temporal_evaluation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-audited IEEE Access revision pipeline"
    )
    parser.add_argument(
        "command",
        choices=[
            "audit",
            "prepare",
            "evaluate",
            "ablation",
            "duplicate-sensitivity",
            "temporal",
            "cold-start",
            "reverse",
            "cluster",
            "figures",
            "all",
        ],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--neural",
        nargs="*",
        default=[],
        choices=["lightfm", "ncf", "lightgcn"],
        help="Neural/hybrid models to include in warm-start evaluation",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config_path = Path(args.config).resolve()
    workspace = config_path.parent.parent
    config = load_config(config_path)
    if args.command == "audit":
        run_audit(config, workspace)
    elif args.command == "prepare":
        run_prepare(config, workspace)
    elif args.command == "evaluate":
        run_evaluation(config, workspace, args.neural)
    elif args.command == "ablation":
        run_feature_ablation(config, workspace)
    elif args.command == "duplicate-sensitivity":
        run_duplicate_sensitivity(config, workspace)
    elif args.command == "temporal":
        run_temporal_evaluation(config, workspace)
    elif args.command == "cold-start":
        run_cold_start_evaluation(config, workspace)
    elif args.command == "reverse":
        run_reverse_evaluation(config, workspace)
    elif args.command == "cluster":
        run_cluster_stability(config, workspace)
    elif args.command == "figures":
        for path in generate_all(config, workspace):
            print(path)
    elif args.command == "all":
        run_audit(config, workspace)
        run_prepare(config, workspace)
        run_feature_ablation(config, workspace)
        run_evaluation(config, workspace, args.neural)
        run_duplicate_sensitivity(config, workspace)
        run_temporal_evaluation(config, workspace)
        run_cold_start_evaluation(config, workspace)
        run_reverse_evaluation(config, workspace)
        run_cluster_stability(config, workspace)
        for path in generate_all(config, workspace):
            print(path)


if __name__ == "__main__":
    main()
