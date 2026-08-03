from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

MOVIES_DTYPE = {
    "movieId": "int32",
    "title": "string",
    "genres": "string",
}

RATINGS_DTYPE = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

NO_GENRES_SENTINEL = "(no genres listed)"


def _check_raw(path: Path) -> None:
    """Friendly error if raw dataset is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            f"See data/raw/README.md for download instructions."
        )


def load_movies_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw movies dataset."""
    path = path or (RAW_DIR / "movies.csv")
    _check_raw(path)

    return pd.read_csv(
        path,
        dtype=MOVIES_DTYPE,
    )


def load_ratings_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw ratings dataset."""
    path = path or (RAW_DIR / "ratings.csv")
    _check_raw(path)

    return pd.read_csv(
        path,
        dtype=RATINGS_DTYPE,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean movies dataset.

    Steps
    -----
    1. Extract release year.
    2. Create genres_list.
    3. Create genres_text.
    """

    out = movies.copy()

    # Extract year from title.
    # Handles:
    # Toy Story (1995)
    # Hamlet (1996) (TV)
    # King Kong (1933) (V)
    out["year"] = (
        out["title"]
        .str.extract(
            r"\((\d{4})\)[^()]*$",
            expand=False,
        )
        .astype("Int32")
    )

    # Convert genres to list.
    out["genres_list"] = (
        out["genres"]
        .fillna(NO_GENRES_SENTINEL)
        .str.split("|")
        .apply(
            lambda xs: [
                g
                for g in xs
                if g and g != NO_GENRES_SENTINEL
            ]
        )
    )

    # Text version for NLP/vectorization.
    out["genres_text"] = (
        out["genres_list"]
        .apply(" ".join)
        .astype("string")
    )

    return out


def clean_ratings(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 50,
    max_iter: int = 10,
) -> pd.DataFrame:
    """
    Clean ratings dataset.

    Steps
    -----
    1. Remove duplicate user-movie ratings.
    2. Remove invalid ratings.
    3. Iteratively filter:
       - active users
       - popular movies
    4. Stop when stable.
    """

    out = ratings.copy()

    # Remove duplicate user-movie pairs.
    out = out.drop_duplicates(
        subset=["userId", "movieId"]
    )

    # Keep valid MovieLens ratings.
    out = out[
(out["rating"] >= 0.5)
        & (out["rating"] <= 5.0)
    ]

    prev_len = -1
    stable = False

    for _ in range(max_iter):

        if len(out) == prev_len:
            stable = True
            break

        prev_len = len(out)

        movie_counts = (
            out.groupby("movieId")
            .size()
        )

        valid_movies = movie_counts[
            movie_counts >= min_movie_ratings
        ].index

        out = out[
            out["movieId"].isin(valid_movies)
        ]

        user_counts = (
            out.groupby("userId")
            .size()
        )

        valid_users = user_counts[
            user_counts >= min_user_ratings
        ].index

        out = out[
            out["userId"].isin(valid_users)
        ]

    if not stable:
        warnings.warn(
            f"clean_ratings: max_iter={max_iter} reached before the filtered "
            "set stabilised. Consider raising max_iter or lowering thresholds.",
            stacklevel=2,
        )

    if out.empty:
        raise ValueError(
            "clean_ratings: all rows were removed by the activity filters "
            f"(min_user_ratings={min_user_ratings}, "
            f"min_movie_ratings={min_movie_ratings}). "
            "Supply a larger dataset or lower the thresholds."
        )

    return out.reset_index(drop=True)


def save_processed(
    movies_clean: pd.DataFrame,
    ratings_cf: pd.DataFrame,
    ratings_content: pd.DataFrame,
) -> None:
    """
    Save processed datasets.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    movies_clean.to_parquet(
        PROCESSED_DIR / "movies_clean.parquet",
        index=False,
    )

    # Cast dtypes explicitly so parquet round-trip matches the data dictionary
    # (int32/int32/float32/int64) even after filter/groupby ops.
    ratings_cf = ratings_cf.astype(RATINGS_DTYPE)
    ratings_content = ratings_content.astype(RATINGS_DTYPE)

    ratings_cf.to_parquet(
        PROCESSED_DIR / "ratings_cf.parquet",
        index=False,
    )

    # Alias kept for plan/docs that still refer to ratings_clean.parquet.
    ratings_cf.to_parquet(
        PROCESSED_DIR / "ratings_clean.parquet",
        index=False,
    )

    ratings_content.to_parquet(
        PROCESSED_DIR / "ratings_content.parquet",
        index=False,
    )


def load_processed() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load processed datasets.
    """
    required = [
        PROCESSED_DIR / "movies_clean.parquet",
        PROCESSED_DIR / "ratings_cf.parquet",
        PROCESSED_DIR / "ratings_content.parquet",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed data files: "
            + ", ".join(missing)
            + ". Run `python scripts/run_pipeline.py` after placing CSVs in data/raw/."
        )

    movies = pd.read_parquet(required[0])
    ratings_cf = pd.read_parquet(required[1])
    ratings_content = pd.read_parquet(required[2])

    # Alias integrity: ratings_clean.parquet must match ratings_cf when present.
    alias = PROCESSED_DIR / "ratings_clean.parquet"
    if alias.exists():
        n_alias = len(pd.read_parquet(alias, columns=["userId"]))
        if n_alias != len(ratings_cf):
            raise ValueError(
                f"ratings_clean.parquet ({n_alias} rows) drifted from "
                f"ratings_cf.parquet ({len(ratings_cf)} rows). "
                "Re-run the pipeline to regenerate both."
            )

    return (
        movies,
        ratings_cf,
        ratings_content,
    )


def run_pipeline() -> None:
    """
    Full preprocessing pipeline.
    """

    movies_raw = load_movies_raw()
    ratings_raw = load_ratings_raw()

    movies_clean = clean_movies(
        movies_raw
    )

    ratings_cf = clean_ratings(
        ratings_raw,
        min_user_ratings=20,
        min_movie_ratings=50,
    )

    ratings_content = clean_ratings(
        ratings_raw,
        min_user_ratings=20,
        min_movie_ratings=5,
    )

    save_processed(
        movies_clean,
        ratings_cf,
        ratings_content,
    )

    print(
        f"movies_clean: {len(movies_clean):,} rows"
    )
    print(
        f"ratings_cf: {len(ratings_cf):,} rows"
    )
    print(
        f"ratings_content: {len(ratings_content):,} rows"
    )


if __name__ == "__main__":
    run_pipeline()