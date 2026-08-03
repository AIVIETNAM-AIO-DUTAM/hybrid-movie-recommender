"""Basic tests — expand as models mature. Owned by QA."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommender_simple import build_movie_stats, recommend_top_movies
from recommender_content import (
    build_content_model,
    recommend_similar_movies,
    genre_overlap_at_k,
)
from recommender_cf import build_cf_model, recommend_for_user
from evaluation import hit_rate_at_k, ndcg_at_k


@pytest.fixture
def tiny_movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy"},
            {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
            {"movieId": 3, "title": "Heat (1995)", "genres": "Action|Crime|Thriller"},
            {"movieId": 4, "title": "Casino (1995)", "genres": "Crime|Drama"},
        ]
    )


@pytest.fixture
def tiny_ratings() -> pd.DataFrame:
    rows = []
    # movie 1 & 2 popular; movie 4 almost no ratings
    for uid in range(1, 30):
        rows.append({"userId": uid, "movieId": 1, "rating": 4.5, "timestamp": 1000 + uid})
        rows.append({"userId": uid, "movieId": 2, "rating": 4.0, "timestamp": 2000 + uid})
        rows.append({"userId": uid, "movieId": 3, "rating": 3.5, "timestamp": 3000 + uid})
    rows.append({"userId": 1, "movieId": 4, "rating": 5.0, "timestamp": 4000})
    return pd.DataFrame(rows)


@pytest.fixture
def cf_movies() -> pd.DataFrame:
    """5 movies; CF is built over movieIds present in ratings."""
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy"},
            {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
            {"movieId": 3, "title": "Heat (1995)", "genres": "Action|Crime|Thriller"},
            {"movieId": 4, "title": "Casino (1995)", "genres": "Crime|Drama"},
            {"movieId": 5, "title": "Sneakers (1992)", "genres": "Crime|Drama|Mystery"},
        ]
    )


@pytest.fixture
def cf_ratings() -> pd.DataFrame:
    """User-movie rating grid designed to leave recommendable candidates.

    All 29 users rate movies 1, 2, 3, 5 (popular). Movie 4 only gets ratings
    from users 5+ so it has enough density but is unseen by user 1 -> user 1
    should get movie 4 (and possibly others) as a recommendation.
    """
    rows = []
    for uid in range(1, 30):
        rows += [
            {"userId": uid, "movieId": 1, "rating": 5.0, "timestamp": 1000 + uid},
            {"userId": uid, "movieId": 2, "rating": 4.5, "timestamp": 2000 + uid},
            {"userId": uid, "movieId": 3, "rating": 4.0, "timestamp": 3000 + uid},
            {"userId": uid, "movieId": 5, "rating": 4.0, "timestamp": 5000 + uid},
        ]
    # Movie 4 rated by users >=5 (NOT by user 1) so user 1 can be recommended it
    for uid in range(5, 30):
        rows.append({"userId": uid, "movieId": 4, "rating": 4.5, "timestamp": 6000 + uid})
    return pd.DataFrame(rows)


@pytest.fixture
def cf_cold_user_ratings(cf_movies) -> pd.DataFrame:
    """Standalone fixture for the F4 (no-liked) cold-start case.

    Self-contained (no `cf_ratings` dependency): a small graph where user 100
    only rates movies below the default `min_rating=4.0`, and there ARE
    recommendable candidates for user 100 — so the only reason
    `recommend_for_user` should bail out is the empty-liked guard, not a
    no-candidate situation. Designed to be wrapped with `pd.concat` against
    cf_ratings when both fixtures are needed together.
    """
    return pd.DataFrame([
        # user 100 — strict ratings only (all below 4.0)
        {"userId": 100, "movieId": 1, "rating": 2.0, "timestamp": 1000},
        {"userId": 100, "movieId": 2, "rating": 3.0, "timestamp": 2000},
        {"userId": 100, "movieId": 3, "rating": 1.5, "timestamp": 3000},
        # user 101 — likes everything so item-sim has signal
        {"userId": 101, "movieId": 1, "rating": 5.0, "timestamp": 4000},
        {"userId": 101, "movieId": 2, "rating": 5.0, "timestamp": 5000},
        {"userId": 101, "movieId": 3, "rating": 5.0, "timestamp": 6000},
        {"userId": 101, "movieId": 4, "rating": 5.0, "timestamp": 7000},
        {"userId": 101, "movieId": 5, "rating": 5.0, "timestamp": 8000},
    ])


def test_simple_top_k(tiny_movies, tiny_ratings):
    stats = build_movie_stats(tiny_ratings)
    out = recommend_top_movies(tiny_movies, stats, top_k=3)
    assert len(out) == 3
    assert {"movieId", "title", "weighted_rating"} <= set(out.columns)


def test_simple_columns_schema(tiny_movies, tiny_ratings):
    """S3: result must expose the full column schema from PDF §3.1.

    Required: movieId, title, genres, avg_rating, num_ratings, weighted_rating.
    Owned by QA (Kiên). Implementation already validates columns; this is a
    guard so future refactors don't drop a column silently.
    """
    stats = build_movie_stats(tiny_ratings)
    out = recommend_top_movies(tiny_movies, stats, top_k=3)
    required = {
        "movieId", "title", "genres",
        "avg_rating", "num_ratings", "weighted_rating",
    }
    assert required <= set(out.columns)


def test_simple_rare_high_rating_not_dominate(tiny_movies, tiny_ratings):
    stats = build_movie_stats(tiny_ratings)
    by_id = stats.set_index("movieId")
    # Movie 4: avg 5.0 nhưng chỉ 1 rating → weighted phải thấp hơn phim phổ biến rating cao
    assert by_id.loc[4, "avg_rating"] == 5.0
    assert by_id.loc[4, "num_ratings"] == 1
    assert by_id.loc[4, "weighted_rating"] < by_id.loc[1, "weighted_rating"]
    out = recommend_top_movies(tiny_movies, stats, top_k=1)
    assert int(out.iloc[0]["movieId"]) == 1


def test_content_excludes_self(tiny_movies):
    model = build_content_model(tiny_movies)
    out = recommend_similar_movies(model, 1, top_k=3)
    assert 1 not in set(out["movieId"])


def test_content_missing_title(tiny_movies):
    model = build_content_model(tiny_movies)
    with pytest.raises(ValueError):
        recommend_similar_movies(model, "Unknown Movie XYZ", top_k=3)


def test_genre_overlap_at_k_all_share(tiny_movies):
    """Spec §6.2: overlap is 1.0 when every top-K shares ≥1 genre with input."""
    # Construct recommendations that all share Adventure with Toy Story.
    recs = pd.DataFrame(
        [
            {"movieId": 2, "title": "Jumanji (1995)", "genres": "Adventure|Children|Fantasy"},
            {"movieId": 99, "title": "Fake Adventure", "genres": "Adventure|Comedy"},
        ]
    )
    input_genres = tiny_movies.loc[tiny_movies["title"] == "Toy Story (1995)", "genres"].iloc[0]
    assert genre_overlap_at_k(recs, input_genres, k=2) == 1.0

    # On the real content model, Toy Story's nearest neighbor (Jumanji) shares genres.
    model = build_content_model(tiny_movies)
    live = recommend_similar_movies(model, "Toy Story (1995)", top_k=1)
    assert genre_overlap_at_k(live, input_genres, k=1) == 1.0


def test_genre_overlap_at_k_none_share():
    recs = pd.DataFrame(
        [
            {"movieId": 10, "title": "A", "genres": "Horror"},
            {"movieId": 11, "title": "B", "genres": "Documentary"},
        ]
    )
    assert genre_overlap_at_k(recs, "Animation|Children", k=2) == 0.0


def test_hr_ndcg_helpers():
    assert hit_rate_at_k([10, 20, 30], 20, k=10) == 1.0
    assert hit_rate_at_k([10, 20, 30], 99, k=10) == 0.0
    assert ndcg_at_k([99, 20], 20, k=10) > 0


# ---------- Collaborative Filtering (F1-F3) ----------

def test_cf_user_in_scope_excludes_seen(cf_movies, cf_ratings):
    """F1: valid user must not get any movie they have already rated."""
    cf = build_cf_model(cf_ratings)
    out = recommend_for_user(cf, cf_movies, user_id=1, top_k=3)
    user_seen = set(cf_ratings.loc[cf_ratings["userId"] == 1, "movieId"])
    assert len(out) > 0
    assert set(out["movieId"]).isdisjoint(user_seen)


def test_cf_unknown_user_raises(cf_movies, cf_ratings):
    """F2: unknown user must signal the app to fall back to Simple."""
    cf = build_cf_model(cf_ratings)
    with pytest.raises(KeyError):
        recommend_for_user(cf, cf_movies, user_id=999_999, top_k=3)


def test_cf_sparse_pipeline_no_oom(cf_movies, cf_ratings):
    """F3: pipeline returns CSR artifacts, no dense materialization."""
    cf = build_cf_model(cf_ratings)
    from scipy.sparse import issparse
    assert issparse(cf.utility)
    assert issparse(cf.item_similarity)
    out = recommend_for_user(cf, cf_movies, user_id=2, top_k=2)
    assert len(out) <= 2


def test_cf_no_liked_movies_raises_valueerror(cf_movies, cf_cold_user_ratings):
    """F4: user with no movie rated >= min_rating signals fallback.

    Spec §3.3 mandates "phim user đã rating cao"; falling back to disliked
    items would bias recommendations. ValueError tells the app to use Simple.

    Uses the standalone `cf_cold_user_ratings` fixture (no `cf_ratings`
    dependency) so the no-liked guard is the only thing being exercised.
    """
    cf = build_cf_model(cf_cold_user_ratings)
    with pytest.raises(ValueError, match="fallback to Simple"):
        recommend_for_user(cf, cf_movies, user_id=100, top_k=5)
