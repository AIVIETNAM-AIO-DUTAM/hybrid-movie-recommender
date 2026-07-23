"""Data pipeline tests — owned by QA (Kiên).

Cases D1-D3 cover Task T10 deliverables. They are SKIPPED until the data
engineer has produced `data/processed/*.parquet`. Once the parquets exist,
unskip them (remove the `pytest.skip()` call) and run:

    pytest tests/test_data_pipeline.py -v

Reference: docs/data-dictionary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROCESSED_MOVIES = ROOT / "data" / "processed" / "movies_clean.parquet"
PROCESSED_RATINGS = ROOT / "data" / "processed" / "ratings_clean.parquet"


def _require_processed():
    """Skip the test if parquet artifacts don't exist yet (Task T10 not done)."""
    if not (PROCESSED_MOVIES.exists() and PROCESSED_RATINGS.exists()):
        pytest.skip(
            "Parquet artifacts missing — run `python scripts/run_pipeline.py` "
            "after Data Engineer finishes T10."
        )


def test_d1_movies_schema():
    """D1: movies_clean has required columns with correct dtypes.

    Spec: docs/data-dictionary.md
    """
    _require_processed()
    # TODO Kiên: implement after T10
    # movies = pd.read_parquet(PROCESSED_MOVIES)
    # required_cols = {"movieId", "title", "year", "genres", "genres_list", "genres_text"}
    # assert required_cols <= set(movies.columns)
    # assert movies["movieId"].is_unique
    pytest.skip("TODO Kiên: implement after T10 delivers parquet")


def test_d2_movies_year_extracted():
    """D2: `year` column is parsed from title (e.g. 'Toy Story (1995)' -> 1995).

    Spec §5.1.
    """
    _require_processed()
    # TODO Kiên
    # movies = pd.read_parquet(PROCESSED_MOVIES)
    # assert movies["year"].notna().mean() > 0.95  # at most 5% missing
    # assert ((movies["year"] >= 1900) & (movies["year"] <= 2030)).mean() > 0.99
    pytest.skip("TODO Kiên: implement after T10 delivers parquet")


def test_d3_ratings_filtered_and_dtype():
    """D3: ratings_clean honors user>=20 / movie>=50 filter and dtype.

    Spec §5.2.
    """
    _require_processed()
    # TODO Kiên
    # ratings = pd.read_parquet(PROCESSED_RATINGS)
    # assert ratings["userId"].dtype == "int32"
    # assert ratings["movieId"].dtype == "int32"
    # assert ratings["rating"].dtype == "float32"
    # assert ratings["timestamp"].dtype == "int64"
    # assert ratings.groupby("userId").size().min() >= 20
    # assert ratings.groupby("movieId").size().min() >= 50
    # assert ratings.duplicated(subset=["userId", "movieId"]).sum() == 0
    pytest.skip("TODO Kiên: implement after T10 delivers parquet")
