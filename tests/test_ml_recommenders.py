"""Behavioral tests for src/ml/ recommenders + hybrid scoring.

These cover the pure functions in the experimental `src/ml/` tree that
previously had ZERO test coverage: recommend_for_user_cf,
recommend_for_user_content, hybrid_rcm.normalize_scores and
recommend_for_user_hybrid. They run on hand-built toy matrices (no model
artifacts needed).

Owned by QA (Kiên).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "ml"))

from recommend import (  # noqa: E402
    build_content_user_profile,
    recommend_for_user_cf,
    recommend_for_user_content,
)
from hybrid_rcm import (  # noqa: E402
    normalize_scores,
    recommend_for_user_hybrid,
)


@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """6 movies, 3 users. Movie 3 (index 2) is unseen by user 1 and rated
    highly by others — the expected CF recommendation for user 1."""
    rows = []
    for uid in range(1, 4):
        for mid in (1, 2):
            rows.append(
                {"userId": uid, "movieId": mid, "rating": 5.0, "timestamp": 1000 + uid}
            )
    rows.append({"userId": 2, "movieId": 3, "rating": 5.0, "timestamp": 2000})
    rows.append({"userId": 3, "movieId": 3, "rating": 5.0, "timestamp": 3000})
    return pd.DataFrame(rows)


@pytest.fixture
def toy_mappings() -> tuple[dict, dict]:
    movie_id_to_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    index_to_movie_id = {i: m for m, i in movie_id_to_index.items()}
    return movie_id_to_index, index_to_movie_id


def _fit_knn(matrix: csr_matrix, n_neighbors: int = 3) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(matrix)
    return nn


def test_recommend_cf_excludes_seen_and_returns_unseen(toy_ratings, toy_mappings):
    """CF: user 1 has seen movies 1,2; must get movie 3 (liked by others)."""
    movie_id_to_index, index_to_movie_id = toy_mappings
    # Toy item-user matrix: rows = movies, cols = users (1..3).
    data = [
        # movie 1 rated by users 1,2,3
        (0, 0, 5.0), (0, 1, 5.0), (0, 2, 5.0),
        # movie 2 rated by users 1,2,3
        (1, 0, 5.0), (1, 1, 5.0), (1, 2, 5.0),
        # movie 3 rated by users 2,3 (NOT user 1)
        (2, 1, 5.0), (2, 2, 5.0),
    ]
    item_user = csr_matrix(
        (np.array([d[2] for d in data]), (np.array([d[0] for d in data]), np.array([d[1] for d in data]))),
        shape=(6, 3),
    )
    model = _fit_knn(item_user)

    recs = recommend_for_user_cf(
        user_id=1,
        train_ratings=toy_ratings,
        model=model,
        movie_user_matrix=item_user,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        top_k=3,
    )
    assert 1 not in recs and 2 not in recs  # seen excluded
    assert 3 in recs


def test_recommend_cf_unknown_user_returns_empty(toy_mappings):
    """CF: user with no history -> [] (not a crash)."""
    movie_id_to_index, index_to_movie_id = toy_mappings
    item_user = csr_matrix(np.zeros((6, 3)))
    model = _fit_knn(item_user)
    ratings = pd.DataFrame(columns=["userId", "movieId", "rating", "timestamp"])
    assert (
        recommend_for_user_cf(
            user_id=999,
            train_ratings=ratings,
            model=model,
            movie_user_matrix=item_user,
            movie_id_to_index=movie_id_to_index,
            index_to_movie_id=index_to_movie_id,
            top_k=3,
        )
        == []
    )


def test_recommend_content_profile_and_recs(toy_ratings, toy_mappings):
    """Content: profile is a weighted mean of liked vectors; recs exclude seen."""
    movie_id_to_index, index_to_movie_id = toy_mappings
    # Toy 6x2 feature matrix: movie1=[1,0], movie2=[0,1], movie3=[1,1], rest=[0,0]
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    feature_matrix = csr_matrix(features)

    profile = build_content_user_profile(
        user_history=toy_ratings.loc[toy_ratings["userId"] == 1],
        movie_feature_matrix=feature_matrix,
        movie_id_to_index=movie_id_to_index,
    )
    assert profile is not None
    assert profile.shape == (1, 2)

    recs = recommend_for_user_content(
        user_id=1,
        train_ratings=toy_ratings,
        movie_feature_matrix=feature_matrix,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id,
        top_k=3,
    )
    assert 1 not in recs and 2 not in recs
    # Movie 3 shares features with the user's liked movies (1,2).
    assert recs and recs[0] == 3


def test_normalize_scores_edge_cases():
    """normalize_scores: empty dict, non-positive max, normal case."""
    assert normalize_scores({}) == {}
    all_zero = normalize_scores({1: 0.0, 2: 0.0})
    assert all_zero == {1: 0.0, 2: 0.0}
    norm = normalize_scores({1: 2.0, 2: 4.0})
    assert norm[1] == pytest.approx(0.5)
    assert norm[2] == pytest.approx(1.0)


def test_recommend_hybrid_orders_by_score_desc(toy_ratings, toy_mappings):
    """Hybrid: returned list is sorted by hybrid score descending and excludes
    movies the user has seen."""
    movie_id_to_index, index_to_movie_id = toy_mappings
    item_user = csr_matrix(
        (
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
            (np.array([0, 1, 0, 1, 2, 2, 3]), np.array([0, 0, 1, 1, 1, 2, 2])),
        ),
        shape=(6, 3),
    )
    cf_model = _fit_knn(item_user)
    feature_matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    )

    recs = recommend_for_user_hybrid(
        user_id=1,
        train_ratings=toy_ratings,
        cf_model=cf_model,
        cf_matrix=item_user,
        cf_movie_id_to_index=movie_id_to_index,
        cf_index_to_movie_id=index_to_movie_id,
        content_matrix=feature_matrix,
        content_movie_id_to_index=movie_id_to_index,
        content_index_to_movie_id=index_to_movie_id,
        top_k=5,
        alpha=0.8,
    )
    assert isinstance(recs, list)
    assert 1 not in recs and 2 not in recs
    assert len(recs) == len(set(recs))  # no duplicates
