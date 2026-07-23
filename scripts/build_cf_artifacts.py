"""CLI entrypoint — build CF artifacts (utility matrix + item similarity).

Owner: 18- Thanh Loan (Pipeline) — Tasks T15 + T17.

What this script does
---------------------
1. Load `data/processed/ratings_clean.parquet`
2. Build sparse utility matrix (user × movie) — Task T15
3. Build sparse item-item cosine similarity (movie × movie) — Task T17
4. Save both to `artifacts/`:
   - utility_matrix.npz
   - item_similarity.npz
   - movie_id_maps.pkl  (user_ids, movie_ids, user_to_row, movie_to_col)
5. Print shapes + density so Loan can sanity-check before hand-off to Tech Lead

Usage
-----
    source .venv/bin/activate
    python scripts/build_cf_artifacts.py

Acceptance (Tasks T15 + T17)
---------------------------
- `artifacts/utility_matrix.npz` exists, sparse, density < 1%
- `artifacts/item_similarity.npz` exists, sparse
- `artifacts/movie_id_maps.pkl` exists with the 4 keys
- `load_cf_artifacts()` returns a working CFModel
- No OOM on the dev machine
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Build and persist CF artifacts.

    TODO Loan: implement across T15 + T17. Suggested steps:
        1. from data_processing import load_processed
        2. from recommender_cf import (
              build_utility_matrix, build_item_similarity,
              build_cf_model, save_cf_artifacts,
           )
        3. movies, ratings = load_processed()
        4. t0 = time.time()
        5. utility, user_ids, movie_ids, u2r, m2c = build_utility_matrix(ratings)
           - T15:csr_matrix((ratings.rating, (user_codes, movie_codes)))
           - print shape + nnz + density
        6. item_sim = build_item_similarity(utility)
           - T17: cosine_similarity(utility.T.tocsr(), dense_output=False)
           - print shape + nnz + density
        7. model = CFModel(utility, item_sim, user_ids, movie_ids, u2r, m2c)
        8. save_cf_artifacts(model)
        9. print wall-clock + artifact sizes
    """
    raise NotImplementedError("TODO Loan: implement T15 + T17 artifact builder")


if __name__ == "__main__":
    main()
