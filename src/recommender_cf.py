"""Item-based Collaborative Filtering. Owned by ML B."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CF_BUILD_META = "cf_build_meta.json"


def _movie_ids_fingerprint(movie_ids: np.ndarray) -> str:
    """Stable SHA1 over sorted movieIds — detects artifact/catalog drift."""
    arr = np.asarray(movie_ids, dtype=np.int64)
    return hashlib.sha1(np.sort(arr).tobytes()).hexdigest()


@dataclass
class CFModel:
    utility: csr_matrix  # user x movie
    item_similarity: csr_matrix  # movie x movie
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_to_row: dict
    movie_to_col: dict


def build_utility_matrix(ratings: pd.DataFrame) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict, dict]:
    if ratings.empty:
        raise ValueError("ratings DataFrame is empty; cannot build utility matrix")
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


def build_item_similarity(
    utility: csr_matrix,
    top_k: int = 100,
    chunk_size: int = 512,
) -> csr_matrix:
    """Item-item cosine similarity, returned as sparse CSR.

    Memory note (spec §11 OOM risk):
    On MovieLens 25M the full (n_movies x n_movies) similarity can OOM.
    This implementation applies the standard kNN-CF mitigation: compute
    cosine in row chunks and keep only the top-`top_k` neighbors per item
    (self excluded). nnz is bounded by O(n_movies * top_k).

    Pass `top_k=0` to keep the full sparse cosine (tests / tiny fixtures).
    """
    from scipy.sparse import lil_matrix

    item_user = utility.T.tocsr()
    n_items = item_user.shape[0]
    if n_items == 0:
        raise ValueError("utility matrix has 0 items; cannot build item similarity")

    if top_k <= 0 or top_k >= n_items:
        return cosine_similarity(item_user, dense_output=False).tocsr()

    # lil_matrix is efficient for incremental row writes; convert to CSR at end.
    sim = lil_matrix((n_items, n_items), dtype=np.float32)
    k = min(top_k, n_items - 1)

    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        # Dense (chunk x n_items) block — chunk_size keeps peak RAM bounded.
        block = cosine_similarity(item_user[start:end], item_user, dense_output=True)
        for local_i, row in enumerate(block):
            global_i = start + local_i
            row[global_i] = -1.0  # exclude self
            if k == 1:
                nn = np.array([int(np.argmax(row))])
            else:
                nn = np.argpartition(-row, k)[:k]
            nn = nn[row[nn] > 0]
            if len(nn) == 0:
                continue
            sim.rows[global_i] = nn.tolist()
            sim.data[global_i] = row[nn].astype(np.float32).tolist()

    return sim.tocsr()


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
    # Detect orphaned movie IDs — present in model but missing from movies DataFrame.
    matched_ids = set(out["movieId"])
    orphaned = [int(mid) for mid in rec_movie_ids if int(mid) not in matched_ids]
    if orphaned:
        warnings.warn(
            f"recommend_for_user: {len(orphaned)} movie ID(s) in CF model not found in "
            f"movies DataFrame and were dropped: {orphaned}",
            UserWarning,
            stacklevel=2,
        )
        out.attrs["orphaned_movie_ids"] = orphaned
    score_map = {int(mid): float(scores[col]) for mid, col in zip(rec_movie_ids, top_cols)}
    out["score"] = out["movieId"].map(score_map)
    out["method"] = "Collaborative Filtering"
    cols = ["movieId", "title", "genres", "score", "method"]
    result = (
        out.sort_values("score", ascending=False).head(top_k)[cols].reset_index(drop=True)
    )
    if orphaned:
        result.attrs["orphaned_movie_ids"] = orphaned
    return result


def save_cf_artifacts(
    model: CFModel,
    prefix: Path | None = None,
    n_ratings: int | None = None,
) -> None:
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
    # Fingerprint so load can detect stale artifacts vs rebuilt parquet.
    meta = {
        "n_users": int(len(model.user_ids)),
        "n_items": int(len(model.movie_ids)),
        "n_ratings": int(
            n_ratings if n_ratings is not None else model.utility.nnz
        ),
        "movie_ids_sha1": _movie_ids_fingerprint(model.movie_ids),
    }
    (prefix / CF_BUILD_META).write_text(json.dumps(meta, indent=2) + "\n")


def load_cf_artifacts(
    prefix: Path | None = None,
    expected_n_ratings: int | None = None,
) -> CFModel:
    prefix = prefix or ARTIFACTS
    required = ["utility_matrix.npz", "item_similarity.npz", "movie_id_maps.pkl"]
    missing = [f for f in required if not (prefix / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"CF artifacts missing in '{prefix}': {missing}. "
            "Run scripts/build_hybrid_artifacts.py to regenerate."
        )
    maps = pd.read_pickle(prefix / "movie_id_maps.pkl")
    utility = load_npz(prefix / "utility_matrix.npz")
    item_similarity = load_npz(prefix / "item_similarity.npz")
    n_users = len(maps["user_ids"])
    n_items = len(maps["movie_ids"])
    if utility.shape != (n_users, n_items):
        raise ValueError(
            f"utility_matrix shape {utility.shape} does not match "
            f"expected ({n_users}, {n_items}) from movie_id_maps.pkl"
        )
    if item_similarity.shape != (n_items, n_items):
        raise ValueError(
            f"item_similarity shape {item_similarity.shape} does not match "
            f"expected ({n_items}, {n_items}) from movie_id_maps.pkl"
        )
    meta_path = prefix / CF_BUILD_META
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if int(meta.get("n_users", -1)) != n_users or int(meta.get("n_items", -1)) != n_items:
            raise ValueError(
                f"{CF_BUILD_META} counts ({meta.get('n_users')}, {meta.get('n_items')}) "
                f"do not match maps ({n_users}, {n_items}). Rebuild CF artifacts."
            )
        expected_fp = meta.get("movie_ids_sha1")
        actual_fp = _movie_ids_fingerprint(maps["movie_ids"])
        if expected_fp and expected_fp != actual_fp:
            raise ValueError(
                f"{CF_BUILD_META} movie_ids fingerprint mismatch "
                f"(meta={expected_fp[:8]}… vs maps={actual_fp[:8]}…). "
                "Rebuild CF artifacts after regenerating processed data."
            )
        # Detect stale artifacts vs rebuilt ratings parquet.
        if expected_n_ratings is not None and "n_ratings" in meta:
            if int(meta["n_ratings"]) != int(expected_n_ratings):
                raise ValueError(
                    f"{CF_BUILD_META} n_ratings={meta['n_ratings']} does not match "
                    f"current ratings ({expected_n_ratings} rows). "
                    "Rebuild CF artifacts after regenerating processed data."
                )
    return CFModel(
        utility=utility,
        item_similarity=item_similarity,
        user_ids=maps["user_ids"],
        movie_ids=maps["movie_ids"],
        user_to_row=maps["user_to_row"],
        movie_to_col=maps["movie_to_col"],
    )
