"""CLI entrypoint — run CF evaluation (HR@10 / NDCG@10).

Owner: 18- Thanh Loan + Hoàng Đức Kiên — Task T19.

What this script does
---------------------
1. Load `data/processed/{movies,ratings}_clean.parquet`
2. Run `evaluation.run_evaluation(ratings, movies, sample_size=200)`
   which does leave-last-out split, builds CF on train, evaluates HR@10/NDCG@10
   on a random sample of 200 users.
3. Print results table to stdout
4. Save CSV to `reports/cf_eval_scores.csv` so QA can chart it in
   `notebooks/05_evaluation_analysis.ipynb`

Usage
-----
    source .venv/bin/activate
    python scripts/run_evaluation.py
    # or with a different sample size:
    python scripts/run_evaluation.py --sample-size 500

Acceptance (Task T19)
---------------------
- Exit code 0
- `reports/cf_eval_scores.csv` exists with columns:
    users_evaluated, HR@10, NDCG@10
- HR@10 > 0 (CF is at least better than random)
- Wall-clock < 5 minutes on the dev machine (with sample_size=200)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORTS_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    """Parse CLI args. TODO Loan: implement (use argparse)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of users to sample for evaluation (default: 200).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="K for HR@K / NDCG@K (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation and save scores.

    TODO Loan + Kiên: implement T19. Suggested steps:
        1. args = parse_args()
        2. from data_processing import load_processed
        3. from evaluation import run_evaluation
        4. movies, ratings = load_processed()
        5. t0 = time.time()
        6. scores = run_evaluation(
              ratings, movies,
              sample_size=args.sample_size,
              top_k=args.top_k,
           )
        7. print scores.to_string(index=False)
        8. print elapsed time
        9. REPORTS_DIR.mkdir(exist_ok=True)
       10. scores.to_csv(REPORTS_DIR / "cf_eval_scores.csv", index=False)
       11. Append row to reports/cf_evaluation.md (date, sample, HR, NDCG)
    """
    raise NotImplementedError("TODO Loan + Kiên: implement T19 evaluation runner")


if __name__ == "__main__":
    main()
