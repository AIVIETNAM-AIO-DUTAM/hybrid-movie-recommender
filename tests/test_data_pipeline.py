"""Data pipeline tests — owned by QA (Kiên).

Cases D1-D3 cover Task T03/T10 deliverables. Prefer full processed
parquets when present; otherwise fall back to the tiny fixtures under
`tests/fixtures/data/processed/` so CI / local machines without MovieLens
can still regression-test schema + dtype contracts.

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
PROCESSED_RATINGS_CF = ROOT / "data" / "processed" / "ratings_cf.parquet"
PROCESSED_RATINGS_CLEAN = ROOT / "data" / "processed" / "ratings_clean.parquet"

FIXTURE_MOVIES = ROOT / "tests" / "fixtures" / "data" / "processed" / "movies_clean.parquet"
FIXTURE_RATINGS = ROOT / "tests" / "fixtures" / "data" / "processed" / "ratings_cf.parquet"


def _movies_path() -> Path:
    if PROCESSED_MOVIES.exists():
        return PROCESSED_MOVIES
    if FIXTURE_MOVIES.exists():
        return FIXTURE_MOVIES
    pytest.skip(
        "Parquet artifacts missing — run `python -c 'from src.data_processing import run_pipeline; run_pipeline()'` "
        "or `python tests/fixtures/build_test_assets.py`."
    )


def _ratings_path() -> Path:
    for path in (PROCESSED_RATINGS_CF, PROCESSED_RATINGS_CLEAN, FIXTURE_RATINGS):
        if path.exists():
            return path
    pytest.skip(
        "Ratings parquet missing — run `python -c 'from src.data_processing import run_pipeline; run_pipeline()'` "
        "or `python tests/fixtures/build_test_assets.py`."
    )


def test_d1_movies_schema():
    """D1: movies_clean has required columns with correct uniqueness."""
    movies = pd.read_parquet(_movies_path())
    required_cols = {"movieId", "title", "year", "genres", "genres_list", "genres_text"}
    assert required_cols <= set(movies.columns)
    assert movies["movieId"].is_unique
    assert len(movies) > 0


def test_d2_movies_year_extracted():
    """D2: `year` column is parsed from title (e.g. 'Toy Story (1995)' -> 1995)."""
    movies = pd.read_parquet(_movies_path())
    year_ok = movies["year"].notna()
    assert year_ok.mean() > 0.95  # at most 5% missing
    valid_range = (movies.loc[year_ok, "year"] >= 1900) & (
        movies.loc[year_ok, "year"] <= 2030
    )
    assert valid_range.mean() > 0.99


def test_d3_ratings_filtered_and_dtype():
    """D3: ratings honor user/movie filters and dtype contract."""
    ratings = pd.read_parquet(_ratings_path())
    assert str(ratings["userId"].dtype) in {"int32", "Int32"}
    assert str(ratings["movieId"].dtype) in {"int32", "Int32"}
    assert str(ratings["rating"].dtype) in {"float32", "Float32"}
    assert str(ratings["timestamp"].dtype) in {"int64", "Int64"}

    # Full processed CF set uses user≥20 / movie≥50. Fixture sets are smaller,
    # so only enforce the filter when we are clearly on the real CF parquet.
    on_full_cf = _ratings_path() in {PROCESSED_RATINGS_CF, PROCESSED_RATINGS_CLEAN}
    if on_full_cf:
        assert ratings.groupby("userId").size().min() >= 20
        assert ratings.groupby("movieId").size().min() >= 50

    assert ratings.duplicated(subset=["userId", "movieId"]).sum() == 0
    assert ((ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)).all()


def test_load_processed_alias_content_drift(tmp_path, monkeypatch):
    """I6: ratings_clean.parquet with same row count but different content
    must be rejected — row-count-only checks miss content drift."""
    import data_processing as dp

    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "A (2000)", "genres": "Drama",
             "year": 2000, "genres_list": ["Drama"], "genres_text": "Drama"}
        ]
    )
    cf = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 100},
            {"userId": 2, "movieId": 1, "rating": 5.0, "timestamp": 200},
        ]
    )
    content = cf.copy()

    monkeypatch.setattr(dp, "PROCESSED_DIR", tmp_path)
    dp.save_processed(movies, cf, content)

    # Corrupt the alias in place: same row count, different rating value.
    alias = tmp_path / "ratings_clean.parquet"
    drifted = cf.copy()
    drifted.loc[0, "rating"] = 1.0
    drifted.to_parquet(alias, index=False)

    with pytest.raises(ValueError, match="content drifted"):
        dp.load_processed()
