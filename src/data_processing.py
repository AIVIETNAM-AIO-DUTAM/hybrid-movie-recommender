"""Load and clean movies/ratings. Owned by Data Engineer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

MOVIES_DTYPE = {"movieId": "int32", "title": "string", "genres": "string"}
RATINGS_DTYPE = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

NO_GENRES_SENTINEL = "(no genres listed)"


def _check_raw(path: Path) -> None:
    """Friendly error pointing to data/raw/README.md when CSV is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. See data/raw/README.md for download "
            f"steps (MovieLens 25M)."
        )


def load_movies_raw(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "movies.csv")
    _check_raw(path)
    return pd.read_csv(path, dtype=MOVIES_DTYPE)


def load_ratings_raw(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "ratings.csv")
    _check_raw(path)
    return pd.read_csv(
        path,
        dtype=RATINGS_DTYPE,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Extract year, split genres, normalize '(no genres listed)' sentinel."""
    out = movies.copy()
    # Match the LAST 4-digit group in parens anywhere in the title.
    # The previous `$`-anchored pattern missed titles like "Foo (2000) (V)"
    # or "Bar (1999) (TV)" used by MovieLens to disambiguate remakes.
    out["year"] = (
        out["title"].str.extract(r"\((\d{4})\)[^()]*$", expand=False).astype("Int32")
    )
    # Normalize genres_list consistently: drop the sentinel in BOTH
    # genres_list and genres_text so downstream consumers (multi-hot
    # encoders, vectorizers) don't treat "(no genres listed)" as a real
    # genre. Aligned with docs/data-dictionary.md.
    out["genres_list"] = (
        out["genres"].fillna(NO_GENRES_SENTINEL)
        .str.split("|")
        .apply(lambda xs: [g for g in xs if g and g != NO_GENRES_SENTINEL])
    )
    out["genres_text"] = out["genres_list"].apply(" ".join)
    return out


def clean_ratings(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50,
    max_iter: int = 10,
) -> pd.DataFrame:
    """Drop duplicates, invalid ratings; iterate filter until stable.

    Filtering user>=20 and movie>=50 in a single pass is non-idempotent:
    after dropping sparse movies, some kept users may now fall below the
    user threshold. We iterate (re-count + filter) until the row count
    stabilizes — typically ≤3 iterations on MovieLens 25M.
    """
    out = ratings.drop_duplicates(subset=["userId", "movieId"]).copy()
    out = out[(out["rating"] >= 0.5) & (out["rating"] <= 5.0)]

    prev_len = -1
    for _ in range(max_iter):
        if len(out) == prev_len:
            break
        prev_len = len(out)
        movie_counts = out.groupby("movieId").size()
        out = out[out["movieId"].isin(
            movie_counts[movie_counts >= min_movie_ratings].index
        )]
        user_counts = out.groupby("userId").size()
        out = out[out["userId"].isin(
            user_counts[user_counts >= min_user_ratings].index
        )]
    return out.reset_index(drop=True)


def save_processed(movies_clean: pd.DataFrame, ratings_clean: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    movies_clean.to_parquet(PROCESSED_DIR / "movies_clean.parquet", index=False)
    ratings_clean.to_parquet(PROCESSED_DIR / "ratings_clean.parquet", index=False)


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_parquet(PROCESSED_DIR / "movies_clean.parquet")
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings_clean.parquet")
    return movies, ratings


def run_pipeline() -> None:
    movies = clean_movies(load_movies_raw())
    ratings = clean_ratings(load_ratings_raw())
    save_processed(movies, ratings)
    print(f"movies_clean: {len(movies):,} rows")
    print(f"ratings_clean: {len(ratings):,} rows")


if __name__ == "__main__":
    run_pipeline()
