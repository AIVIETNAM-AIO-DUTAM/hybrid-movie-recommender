from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DATA_DIR = ROOT / "tests" / "fixtures" / "data" / "processed"


def test_d1_movies_schema():
    movies = pd.read_parquet(FIXTURE_DATA_DIR / "movies_clean.parquet")

    required_columns = {
        "movieId",
        "title",
        "year",
        "genres",
        "genres_list",
        "genres_text",
    }
    assert required_columns <= set(movies.columns)
    assert not movies.empty
    assert movies["movieId"].is_unique
    assert movies["movieId"].notna().all()
    assert movies["title"].str.len().gt(0).all()


def test_d2_movies_year_extracted():
    movies = pd.read_parquet(FIXTURE_DATA_DIR / "movies_clean.parquet")

    valid_years = movies["year"].between(1800, 2100)
    assert valid_years.mean() >= 0.95


def test_d3_ratings_filtered_and_dtype():
    movies = pd.read_parquet(FIXTURE_DATA_DIR / "movies_clean.parquet")
    ratings = pd.read_parquet(FIXTURE_DATA_DIR / "ratings_cf.parquet")
    train_ratings = pd.read_parquet(
        FIXTURE_DATA_DIR / "rating_cf_train.parquet"
    )

    required_columns = {"userId", "movieId", "rating", "timestamp"}
    assert required_columns <= set(ratings.columns)
    assert required_columns <= set(train_ratings.columns)
    assert not ratings.empty
    assert not train_ratings.empty

    for column in ["userId", "movieId", "timestamp"]:
        assert pd.api.types.is_integer_dtype(ratings[column])
        assert pd.api.types.is_integer_dtype(train_ratings[column])

    assert pd.api.types.is_float_dtype(ratings["rating"])
    assert pd.api.types.is_float_dtype(train_ratings["rating"])
    assert ratings["rating"].between(0.5, 5.0).all()
    assert train_ratings["rating"].between(0.5, 5.0).all()
    assert not ratings.duplicated(["userId", "movieId"]).any()

    known_movie_ids = set(movies["movieId"].astype(int))
    assert set(ratings["movieId"].astype(int)) <= known_movie_ids
    assert set(train_ratings["movieId"].astype(int)) <= set(
        ratings["movieId"].astype(int)
    )
