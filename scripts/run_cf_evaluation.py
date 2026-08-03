"""Reproduce CF HR@10 / NDCG@10 cited in reports/final_report.md §5.1 and reports/cf_evaluation.md.

Wraps `src.evaluation.run_evaluation` (leave-last-out, item-based CF on `ratings_cf.parquet`)
and writes results to `evaluation/cf_eval_scores.csv`.

Usage
-----
    source .venv/bin/activate
    python scripts/run_cf_evaluation.py                      # default: 200 users, top_k=10
    python scripts/run_cf_evaluation.py --sample-size 500
    python scripts/run_cf_evaluation.py --sample-size 0      # full eligible population (slow)

Output
------
- evaluation/cf_eval_scores.csv   (one-row summary: users_evaluated, HR@10, NDCG@10, HR@10_all, NDCG@10_all)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # evaluation.py does `from recommender_cf import`

DATA_PROCESSED = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "evaluation"

RATINGS_PATH = DATA_PROCESSED / "ratings_cf.parquet"
MOVIES_PATH = DATA_PROCESSED / "movies_clean.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of eligible users to sample (default: 200). Pass 0 for full population.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_size: int | None = None if args.sample_size == 0 else args.sample_size

    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RATINGS_PATH}. Run "
            "`python -c 'from src.data_processing import run_pipeline; run_pipeline()'` first."
        )
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MOVIES_PATH}. Run "
            "`python -c 'from src.data_processing import run_pipeline; run_pipeline()'` first."
        )

    ratings = pd.read_parquet(RATINGS_PATH)
    movies = pd.read_parquet(MOVIES_PATH)

    from src.evaluation import run_evaluation

    summary = run_evaluation(
        ratings=ratings,
        movies=movies,
        sample_size=sample_size,
        top_k=args.top_k,
        min_rating=args.min_rating,
        seed=args.seed,
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_DIR / "cf_eval_scores.csv"
    summary.to_csv(summary_path, index=False)

    print("=" * 60)
    print("CF evaluation done.")
    print(summary.to_string(index=False))
    print(f"\nWrote: {summary_path}")


if __name__ == "__main__":
    main()
