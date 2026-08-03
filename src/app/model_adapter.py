from __future__ import annotations

import sys
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import load_npz


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
SRC_ML_DIR = SRC_DIR / "ml"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_ML_DIR))

from hybrid_rcm import (
    get_cf_candidate_scores,
    get_content_candidate_scores,
    normalize_scores,
)


DATA_DIR = Path(
    os.getenv("REC_DATA_DIR", ROOT / "data" / "processed")
)
MOVIES_PATH = DATA_DIR / "movies_clean.parquet"
RATINGS_PATH = DATA_DIR / "ratings_cf.parquet"
TRAIN_RATINGS_PATH = DATA_DIR / "rating_cf_train.parquet"

MODEL_DIR = Path(os.getenv("REC_MODEL_DIR", ROOT / "model"))
CF_MODEL_DIR = MODEL_DIR / "knn_cf"
CONTENT_MODEL_DIR = MODEL_DIR / "knn_content"

MODEL_NAME = "Hybrid Recommender"
MODEL_OPTIONS = [MODEL_NAME]
SUPPORTS_TOP_K = True

DEFAULT_ALPHA = 0.8
DEFAULT_NEIGHBORS_PER_MOVIE = 50
DEFAULT_CONTENT_CANDIDATE_LIMIT = 500
DEFAULT_POSITIVE_THRESHOLD = 4.0


@st.cache_data(show_spinner=True)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists():
        missing = [
            str(path)
            for path in [MOVIES_PATH, RATINGS_PATH]
            if not path.exists()
        ]
        raise FileNotFoundError(
            "Missing processed data files: " + ", ".join(missing)
        )

    movies = pd.read_parquet(MOVIES_PATH)
    ratings = pd.read_parquet(RATINGS_PATH)

    movies["movieId"] = movies["movieId"].astype(int)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return movies, ratings


@st.cache_data(show_spinner=True)
def load_train_ratings() -> pd.DataFrame:
    if not TRAIN_RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"Missing train ratings file: {TRAIN_RATINGS_PATH}"
        )

    ratings = pd.read_parquet(TRAIN_RATINGS_PATH)
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings


def normalize_mappings(mappings: dict) -> tuple[dict[int, int], dict[int, int]]:
    if (
        "movie_id_to_index" in mappings
        and "index_to_movie_id" in mappings
    ):
        movie_id_to_index = {
            int(movie_id): int(index)
            for movie_id, index in mappings["movie_id_to_index"].items()
        }
        index_to_movie_id = {
            int(index): int(movie_id)
            for index, movie_id in mappings["index_to_movie_id"].items()
        }
        return movie_id_to_index, index_to_movie_id

    if "movie_ids" in mappings:
        movie_ids = mappings["movie_ids"]
        movie_id_to_index = {
            int(movie_id): int(index)
            for index, movie_id in enumerate(movie_ids)
        }
        index_to_movie_id = {
            int(index): int(movie_id)
            for index, movie_id in enumerate(movie_ids)
        }
        return movie_id_to_index, index_to_movie_id

    raise ValueError("Không tìm thấy movie mapping trong artifact.")


def check_artifact(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")


@st.cache_resource(show_spinner=True)
def load_hybrid_artifacts() -> dict:
    cf_model_path = CF_MODEL_DIR / "knn_cf_model.joblib"
    cf_matrix_path = CF_MODEL_DIR / "movie_user_matrix.npz"
    cf_mappings_path = CF_MODEL_DIR / "cf_mappings.joblib"

    content_matrix_path = CONTENT_MODEL_DIR / "movie_feature_matrix.npz"
    content_mappings_path = CONTENT_MODEL_DIR / "content_mappings.joblib"

    for path in [
        cf_model_path,
        cf_matrix_path,
        cf_mappings_path,
        content_matrix_path,
        content_mappings_path,
    ]:
        check_artifact(path)

    cf_mappings = joblib.load(cf_mappings_path)
    content_mappings = joblib.load(content_mappings_path)

    cf_movie_id_to_index, cf_index_to_movie_id = normalize_mappings(
        cf_mappings
    )
    content_movie_id_to_index, content_index_to_movie_id = normalize_mappings(
        content_mappings
    )

    return {
        "cf_model": joblib.load(cf_model_path),
        "cf_matrix": load_npz(cf_matrix_path),
        "cf_movie_id_to_index": cf_movie_id_to_index,
        "cf_index_to_movie_id": cf_index_to_movie_id,
        "content_matrix": load_npz(content_matrix_path),
        "content_movie_id_to_index": content_movie_id_to_index,
        "content_index_to_movie_id": content_index_to_movie_id,
    }


@st.cache_data(show_spinner=True)
def get_user_options(
    ratings: pd.DataFrame,
    limit: int = 5000,
) -> list[int]:
    user_ids = (
        ratings["userId"]
        .drop_duplicates()
        .astype(int)
        .sort_values()
        .head(limit)
        .tolist()
    )
    return [int(user_id) for user_id in user_ids]


def _simple_fallback_rows(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Cold-start fallback: top movies by weighted rating (Simple recommender).

    Mirrors the 3-tab app (src/app.py `_fallback_simple`) so the hybrid UI
    never shows an empty panel when a user has no CF/content signal.
    """
    from recommender_simple import build_movie_stats, recommend_top_movies

    stats = build_movie_stats(ratings)
    simple = recommend_top_movies(movies, stats, top_k=top_k)
    simple = simple.rename(
        columns={
            "avg_rating": "rating",
            "weighted_rating": "model_score",
        }
    )
    simple["model_score"] = simple["model_score"].astype(float)
    simple = simple[["movieId", "title", "genres", "rating", "num_ratings", "model_score"]]
    simple.insert(0, "rank", range(1, len(simple) + 1))
    return simple.reset_index(drop=True)


def predict(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    top_k: int = 10,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Return recommendation rows expected by the Streamlit UI.

    This is the main integration point for the final model. Keep the output
    columns stable:
    rank, movieId, title, genres, rating, num_ratings, model_score.
    """

    if model_name not in MODEL_OPTIONS:
        raise ValueError(f"Unsupported model: {model_name}")

    train_ratings = load_train_ratings()
    artifacts = load_hybrid_artifacts()

    cf_scores = get_cf_candidate_scores(
        user_id=user_id,
        train_ratings=train_ratings,
        model=artifacts["cf_model"],
        movie_user_matrix=artifacts["cf_matrix"],
        movie_id_to_index=artifacts["cf_movie_id_to_index"],
        index_to_movie_id=artifacts["cf_index_to_movie_id"],
        neighbors_per_movie=DEFAULT_NEIGHBORS_PER_MOVIE,
        positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
    )
    content_scores = get_content_candidate_scores(
        user_id=user_id,
        train_ratings=train_ratings,
        movie_feature_matrix=artifacts["content_matrix"],
        movie_id_to_index=artifacts["content_movie_id_to_index"],
        index_to_movie_id=artifacts["content_index_to_movie_id"],
        positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
        candidate_limit=DEFAULT_CONTENT_CANDIDATE_LIMIT,
    )

    cf_scores = normalize_scores(cf_scores)
    content_scores = normalize_scores(content_scores)

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId"],
    ]
    seen_movie_ids = set(user_history["movieId"].astype(int).tolist())

    hybrid_scores = {}
    for movie_id in set(cf_scores) | set(content_scores):
        if movie_id in seen_movie_ids:
            continue

        hybrid_scores[int(movie_id)] = (
            DEFAULT_ALPHA * cf_scores.get(movie_id, 0.0)
            + (1.0 - DEFAULT_ALPHA) * content_scores.get(movie_id, 0.0)
        )

    ranked_movie_ids = [
        movie_id
        for movie_id, _ in sorted(
            hybrid_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
    ]

    if not ranked_movie_ids:
        # Cold-start: user has no hybrid signal (empty history, nothing liked,
        # or zero candidates). Fall back to Simple top-K so the UI never shows
        # an empty panel — mirrors the 3-tab app's fallback behavior.
        return _simple_fallback_rows(movies, ratings, top_k)

    movie_stats = (
        ratings.groupby("movieId")
        .agg(
            rating=("rating", "mean"),
            num_ratings=("rating", "count"),
        )
        .reset_index()
    )

    output = (
        pd.DataFrame(
            {
                "movieId": ranked_movie_ids,
                "model_score": [
                    hybrid_scores[movie_id]
                    for movie_id in ranked_movie_ids
                ],
            }
        )
        .merge(movies, on="movieId", how="left")
        .merge(movie_stats, on="movieId", how="left")
    )
    # Hybrid movie IDs can come from CF/content artifacts built on a different
    # parquet version. Left-merge then silently displays NaN title/genres —
    # drop orphan rows and surface the count (mirrors recommender_cf's orphan
    # warning instead of showing blank rows).
    orphan_mask = output["title"].isna()
    n_orphans = int(orphan_mask.sum())
    if n_orphans:
        import warnings

        warnings.warn(
            f"predict: dropped {n_orphans} hybrid candidate movieId(s) "
            "not present in movies_clean.parquet",
            UserWarning,
            stacklevel=2,
        )
        output = output[~orphan_mask]
    output.insert(0, "rank", range(1, len(output) + 1))
    output["rating"] = output["rating"].fillna(0.0)
    output["num_ratings"] = output["num_ratings"].fillna(0).astype(int)

    expected_columns = [
        "rank",
        "movieId",
        "title",
        "genres",
        "rating",
        "num_ratings",
        "model_score",
    ]
    output = output[expected_columns]
    return output.reset_index(drop=True)


def get_user_context(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    history = ratings.loc[
        ratings["userId"] == user_id,
        ["movieId", "rating", "timestamp"],
    ].copy()

    if history.empty:
        return history

    movie_columns = ["movieId", "title", "genres"]
    if "year" in movies.columns:
        movie_columns.append("year")

    history = history.merge(
        movies[movie_columns],
        on="movieId",
        how="left",
    )
    history["rated_at"] = pd.to_datetime(
        history["timestamp"],
        unit="s",
        errors="coerce",
    )
    return history.sort_values(
        "timestamp",
        ascending=False,
    ).reset_index(drop=True)
