"""CLI entrypoint — run the full data pipeline.

Owner: Trần Hoàng Minh Tâm (Data Engineer) — Task T10.

What this script does
---------------------
1. Read raw CSVs from `data/raw/` (movies.csv, ratings.csv)
2. Clean them via `data_processing.clean_movies` / `clean_ratings`
3. Save parquet to `data/processed/`
4. Print row counts (before/after filter) so the data engineer can copy
   these numbers into `reports/eda_summary.md`

Usage
-----
    source .venv/bin/activate
    python scripts/run_pipeline.py

Output
------
    data/processed/movies_clean.parquet
    data/processed/ratings_clean.parquet

Acceptance (Task T10)
---------------------
- Exit code 0 on MovieLens 25M
- Both parquet files exist after run
- stdout shows: raw row counts, filtered row counts, % retained
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Run the data pipeline end-to-end and print summary stats.

    TODO Tâm: implement. Suggested steps:
        1. from data_processing import (
              load_movies_raw, load_ratings_raw,
              clean_movies, clean_ratings, save_processed,
           )
        2. t0 = time.time()
        3. movies_raw = load_movies_raw(); n_movies_raw = len(movies_raw)
        4. ratings_raw = load_ratings_raw(); n_ratings_raw = len(ratings_raw)
        5. movies_clean = clean_movies(movies_raw)
        6. ratings_clean = clean_ratings(ratings_raw)
        7. save_processed(movies_clean, ratings_clean)
        8. Print summary table:
              raw movies / clean movies
              raw ratings / clean ratings (% retained)
              wall-clock seconds
        9. Optionally write summary to reports/eda_summary.md (append)
    """
    raise NotImplementedError("TODO Tâm: implement T10 pipeline runner")


if __name__ == "__main__":
    main()
