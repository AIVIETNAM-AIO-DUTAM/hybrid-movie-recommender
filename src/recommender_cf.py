"""Item-based Collaborative Filtering. Owned by ML B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@dataclass
class CFModel:
    utility: csr_matrix  # user x movie
    item_similarity: csr_matrix  # movie x movie
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_row: dict
    movie_to_col: dict


def build_utility_matrix(ratings: pd.DataFrame) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict, dict]:
    user_cats = ratings["userId"].astype("category")
    movie_cats = ratings["movieId"].astype("category")
    user_ids = user_cats.cat.categories.to_numpy()
    movie_ids = movie_cats.cat.categories.to_numpy()
    rows = user_cats.cat.codes.to_numpy()
    cols = movie_cats.cat.codes.to_numpy()
    data = ratings["rating"].to_numpy(dtype=np.float32)
    utility = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    user_to_row = {int(u): i for i, u in enumerate(user_ids)}
    movie_to_col = {int(m): i for i, m in enumerate(movie_ids)}
    return utility, user_ids, movie_ids, user_to_row, movie_to_col


def build_item_similarity(utility: csr_matrix) -> csr_matrix:
    """Item-item cosine similarity, returned as sparse CSR.

    Memory note (spec §11 OOM risk):
    On MovieLens 25M (~62k movies, ~162k users) the cosine similarity matrix
    is (62k x 62k). Even with `dense_output=False`, scipy CSR overhead is
    ~16 bytes/nnz vs 8 bytes for dense float64 — if the matrix is more than
    ~50% dense the sparse form is LARGER than dense. Empirically item
    similarity can reach 30-80% density, risking OOM on machines with <32 GB.

    Mitigations (apply at artifact-build time in scripts/build_cf_artifacts.py):
      1. Pre-filter ratings harder before build_utility_matrix:
         min_user=50, min_movie=200 (drops long-tail items that create
         high-density noise).
      2. Threshold the similarity matrix: keep only top-K neighbors per item
         (e.g. `argpartition` per row, zero out the rest) — this is the
         standard kNN-CF pattern and bounds nnz to O(n_movies * k).
      3. If still OOM: sample users (e.g. random 50k users) for the utility
         matrix; CF still works, just less coverage.
      4. Last resort: chunk the similarity computation (process N items at a
         time) and `hstack` the sparse blocks.

    The current implementation keeps the simple `cosine_similarity(utility.T)`
    form because it is correct and matches the spec pseudocode (§13.4). Loan
    should apply mitigation #1 and #2 on the 25M dataset if memory pressure
    appears during T07.
    """
    item_user = utility.T.tocsr()
    return cosine_similarity(item_user, dense_output=False)


def build_cf_model(ratings: pd.DataFrame) -> CFModel:
    utility, user_ids, movie_ids, user_to_row, movie_to_col = build_utility_matrix(ratings)
    item_sim = build_item_similarity(utility)
    return CFModel(
        utility=utility,
        item_similarity=item_sim,
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_to_row=user_to_row,
        movie_to_col=movie_to_col,
    )


def recommend_for_user(
    model: CFModel,
    movies: pd.DataFrame,
    user_id: int,
    top_k: int = 10,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Personalized top-K.

    Raises:
        KeyError: if user unknown — caller should fall back to Simple.
        ValueError: if user has no liked items OR no CF candidates — caller
            should fall back to Simple (per spec §11 cold-start risk).
    """
    if user_id not in model.user_to_row:
        raise KeyError(f"unknown userId: {user_id}")

    urow = model.user_to_row[user_id]
    user_ratings = model.utility.getrow(urow).tocoo()
    liked_cols = [
        c for c, r in zip(user_ratings.col, user_ratings.data) if r >= min_rating
    ]
    # Spec §3.3: "lấy các phim user đã rating cao". When the user has no
    # "liked" items, falling back to ALL rated movies would bias toward
    # disliked items. Instead, signal the caller to fall back to Simple.
    if not liked_cols:
        raise ValueError(
            f"user {user_id} has no movies rated >= {min_rating}; "
            f"fallback to Simple"
        )

    seen = set(user_ratings.col)

    # Sparse matmul: (1 x n_liked) @ (n_liked x n_movies) -> score per movie.
    # Avoids per-liked-movie todense() calls which are slow and RAM-heavy.
    n_movies = model.utility.shape[1]
    liked_mask = csr_matrix(
        (np.ones(len(liked_cols), dtype=np.float32), (np.zeros(len(liked_cols)), liked_cols)),
        shape=(1, n_movies),
    )
    scores = (liked_mask @ model.item_similarity).toarray().ravel()

    for c in seen:
        scores[c] = -np.inf

    # Candidates = movies the user has not seen AND that have a non-zero score.
    unseen = np.setdiff1d(np.arange(n_movies), list(seen), assume_unique=False)
    # Filter to candidates with positive similarity mass to avoid returning
    # pure noise when the user has liked nothing similar.
    unseen = unseen[scores[unseen] > 0]
    if len(unseen) == 0:
        # No candidate movies. Caller (app.py) should catch ValueError and
        # fall back to Simple per spec §11.
        raise ValueError(
            f"no CF candidates for user {user_id}; fallback to Simple"
        )

    k = min(top_k, len(unseen))
    top_cols = np.argpartition(-scores[unseen], k - 1)[:k]
    # argsort(-scores) already returns indices in descending-score order;
    # do NOT append [::-1] (would flip to worst-first).
    top_cols = unseen[np.argsort(-scores[unseen][top_cols])]

    rec_movie_ids = model.movie_ids[top_cols]
    out = movies[movies["movieId"].isin(rec_movie_ids)].copy()
    score_map = {int(mid): float(scores[col]) for mid, col in zip(rec_movie_ids, top_cols)}
    out["score"] = out["movieId"].map(score_map)
    out["method"] = "Collaborative Filtering"
    cols = ["movieId", "title", "genres", "score", "method"]
    return out.sort_values("score", ascending=False).head(top_k)[cols].reset_index(drop=True)


def save_cf_artifacts(model: CFModel, prefix: Path | None = None) -> None:
    prefix = prefix or ARTIFACTS
    prefix.mkdir(parents=True, exist_ok=True)
    save_npz(prefix / "utility_matrix.npz", model.utility)
    save_npz(prefix / "item_similarity.npz", model.item_similarity)
    pd.to_pickle(
        {
            "user_ids": model.user_ids,
            "movie_ids": model.movie_ids,
            "user_to_row": model.user_to_row,
            "movie_to_col": model.movie_to_col,
        },
        prefix / "movie_id_maps.pkl",
    )


def load_cf_artifacts(prefix: Path | None = None) -> CFModel:
    prefix = prefix or ARTIFACTS
    maps = pd.read_pickle(prefix / "movie_id_maps.pkl")
    return CFModel(
        utility=load_npz(prefix / "utility_matrix.npz"),
        item_similarity=load_npz(prefix / "item_similarity.npz"),
        user_ids=maps["user_ids"],
        movie_ids=maps["movie_ids"],
        user_to_row=maps["user_to_row"],
        movie_to_col=maps["movie_to_col"],
    )
