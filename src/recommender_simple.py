"""Simple Recommender — weighted rating. Owned by ML A."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def build_movie_stats(ratings: pd.DataFrame, m_quantile: float = 0.80) -> pd.DataFrame:
    """Compute avg_rating, num_ratings, weighted_rating per movieId."""
    if ratings.empty:
        raise ValueError("ratings DataFrame is empty; cannot compute movie stats")
    if ratings["rating"].isna().all():
        raise ValueError("ratings column is all-NaN; cannot compute movie stats")
    stats = (
        ratings.groupby("movieId")
        .agg(avg_rating=("rating", "mean"), num_ratings=("rating", "count"))
        .reset_index()
    )
    c = float(ratings["rating"].mean())
    m = float(stats["num_ratings"].quantile(m_quantile))
    v = stats["num_ratings"]
    r = stats["avg_rating"]
    stats["weighted_rating"] = (v / (v + m)) * r + (m / (v + m)) * c
    stats.attrs["C"] = c
    stats.attrs["m"] = m
    return stats


def recommend_top_movies(
    movies: pd.DataFrame,
    movie_stats: pd.DataFrame,
    top_k: int = 10,
    genre: Optional[str] = None,
) -> pd.DataFrame:
    """Return top-K movies by weighted_rating. Optional genre contains-filter.

    Follows IMDb-style rule (spec §3.1): only rank movies whose `num_ratings`
    is at least `m` (stored on `movie_stats.attrs["m"]`). If that filter would
    remove everything (small dev fixtures), we fall back to ranking all rows
    so the recommender still returns a meaningful answer instead of crashing.
    """
    df = movie_stats.merge(movies, on="movieId", how="inner")
    if genre:
        df = df[df["genres"].fillna("").str.contains(genre, case=False, regex=False)]

    m = movie_stats.attrs.get("m")
    if m is not None:
        qualified = df[df["num_ratings"] >= m]
        if not qualified.empty:
            df = qualified

    cols = [
        "movieId",
        "title",
        "genres",
        "avg_rating",
        "num_ratings",
        "weighted_rating",
    ]
    # Secondary keys (num_ratings desc, movieId asc) ensure deterministic
    # output when two movies share the same weighted_rating. Without this
    # the order is decided by pandas' unstable quicksort and varies run to run.
    return (
        df.sort_values(
            ["weighted_rating", "num_ratings", "movieId"],
            ascending=[False, False, True],
        )
        .head(top_k)[cols]
        .reset_index(drop=True)
    )
