"""Edge-case guards added after parallel codebase review."""

from __future__ import annotations

import ast
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing import clean_ratings, load_processed  # noqa: E402
from evaluation import leave_last_out_split  # noqa: E402
from evaluation import evaluate, prepare_eval  # noqa: E402
from recommender_cf import (  # noqa: E402
    build_cf_model,
    build_utility_matrix,
    load_cf_artifacts,
    recommend_for_user,
    save_cf_artifacts,
)
from recommender_content import (  # noqa: E402
    build_content_model,
    recommend_similar_movies,
)
from recommender_simple import build_movie_stats  # noqa: E402


def test_cf_artifacts_ready_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "demo_app", ROOT / "src" / "app.py"
    )
    # Don't exec Streamlit module (set_page_config). Test pure helper via source.
    src = (ROOT / "src" / "app.py").read_text()
    assert "def _cf_artifacts_ready" in src
    assert "CF_ARTIFACT_FILES" in src


def test_load_cf_artifacts_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        load_cf_artifacts(tmp_path)


def test_cf_build_meta_fingerprint_roundtrip(tmp_path, cf_model_mini):
    save_cf_artifacts(cf_model_mini, prefix=tmp_path, n_ratings=99)
    meta_path = tmp_path / "cf_build_meta.json"
    assert meta_path.exists()
    loaded = load_cf_artifacts(tmp_path)
    assert len(loaded.movie_ids) == len(cf_model_mini.movie_ids)

    # Corrupt fingerprint → refuse load
    meta = json.loads(meta_path.read_text())
    meta["movie_ids_sha1"] = "0" * 40
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="fingerprint"):
        load_cf_artifacts(tmp_path)


def test_cf_load_missing_meta_raises(tmp_path, cf_model_mini):
    """I8: artifacts without cf_build_meta.json must be refused, not silently
    served with every provenance check skipped."""
    save_cf_artifacts(cf_model_mini, prefix=tmp_path, n_ratings=99)
    (tmp_path / "cf_build_meta.json").unlink()
    with pytest.raises(ValueError, match="cf_build_meta.json is missing"):
        load_cf_artifacts(tmp_path)


def test_cf_build_meta_n_ratings_drift(tmp_path, cf_model_mini):
    save_cf_artifacts(cf_model_mini, prefix=tmp_path, n_ratings=99)
    with pytest.raises(ValueError, match="n_ratings"):
        load_cf_artifacts(tmp_path, expected_n_ratings=100)


def test_simple_all_nan_ratings_raises():
    df = pd.DataFrame(
        {"userId": [1], "movieId": [1], "rating": [float("nan")], "timestamp": [1]}
    )
    with pytest.raises(ValueError, match="all-NaN"):
        build_movie_stats(df)


def test_build_utility_empty_ratings():
    empty = pd.DataFrame(columns=["userId", "movieId", "rating"])
    with pytest.raises(ValueError, match="empty"):
        build_utility_matrix(empty)


def test_build_utility_nan_ids_raise():
    """I3: NaN userId/movieId/rating must raise early, not create phantom
    category rows or crash later inside int(np.nan)."""
    bad = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": 4.0},
            {"userId": float("nan"), "movieId": 2, "rating": 4.0},
        ]
    )
    with pytest.raises(ValueError, match="NaN"):
        build_utility_matrix(bad)

    bad_movie = pd.DataFrame(
        [
            {"userId": 1, "movieId": float("nan"), "rating": 4.0},
            {"userId": 1, "movieId": 2, "rating": 4.0},
        ]
    )
    with pytest.raises(ValueError, match="NaN"):
        build_utility_matrix(bad_movie)

    bad_rating = pd.DataFrame(
        [
            {"userId": 1, "movieId": 1, "rating": float("nan")},
            {"userId": 1, "movieId": 2, "rating": 4.0},
        ]
    )
    with pytest.raises(ValueError, match="NaN"):
        build_utility_matrix(bad_rating)


def test_recommend_orphaned_movie_ids_warns(cf_movies_mini, cf_model_mini):
    movies_partial = cf_movies_mini[cf_movies_mini["movieId"] != 4]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = recommend_for_user(cf_model_mini, movies_partial, user_id=1, top_k=5)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert out.attrs.get("orphaned_movie_ids")


def test_simple_empty_ratings_raises():
    with pytest.raises(ValueError, match="empty"):
        build_movie_stats(pd.DataFrame(columns=["userId", "movieId", "rating"]))


def test_content_empty_vocabulary_raises():
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "A", "genres": "(no genres listed)"},
            {"movieId": 2, "title": "B", "genres": "(no genres listed)"},
        ]
    )
    with pytest.raises(ValueError, match="vocabulary"):
        build_content_model(movies)


def test_content_top_k_does_not_leak_self():
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "A (1995)", "genres": "Action"},
            {"movieId": 2, "title": "B (1996)", "genres": "Action"},
        ]
    )
    model = build_content_model(movies)
    out = recommend_similar_movies(model, 1, top_k=10)
    assert 1 not in set(out["movieId"].tolist())
    assert len(out) == 1


def test_content_single_movie_returns_empty():
    movies = pd.DataFrame(
        [{"movieId": 1, "title": "Solo (1995)", "genres": "Drama"}]
    )
    model = build_content_model(movies)
    out = recommend_similar_movies(model, 1, top_k=5)
    assert out.empty


def test_clean_ratings_empty_raises():
    tiny = pd.DataFrame(
        [{"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1}]
    )
    with pytest.raises(ValueError, match="all rows were removed"):
        clean_ratings(tiny, min_user_ratings=20, min_movie_ratings=50)


def test_clean_ratings_duplicate_keeps_most_recent():
    """I7: duplicate (user, movie) with unsorted rows must keep the newest
    timestamp, not the first row encountered in the CSV."""
    rows = [
        {"userId": 1, "movieId": 1, "rating": 2.0, "timestamp": 100},
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 300},
        {"userId": 1, "movieId": 1, "rating": 3.0, "timestamp": 200},
    ]
    # Reversed input order to prove we don't just keep "first row".
    out = clean_ratings(
        pd.DataFrame(list(reversed(rows))),
        min_user_ratings=1,
        min_movie_ratings=1,
    )
    assert len(out) == 1
    assert out.iloc[0]["rating"] == 5.0
    assert out.iloc[0]["timestamp"] == 300


def test_load_processed_missing(monkeypatch, tmp_path):
    import data_processing as dp

    monkeypatch.setattr(dp, "PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing processed"):
        load_processed()


def test_leave_last_out_timestamp_ties_deterministic():
    rows = [
        {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 100},
        {"userId": 1, "movieId": 20, "rating": 5.0, "timestamp": 100},
        {"userId": 1, "movieId": 30, "rating": 3.0, "timestamp": 50},
    ]
    a = pd.DataFrame(rows)
    b = pd.DataFrame(list(reversed(rows)))
    _, test_a = leave_last_out_split(a)
    _, test_b = leave_last_out_split(b)
    assert int(test_a.iloc[0]["movieId"]) == int(test_b.iloc[0]["movieId"])


def test_evaluate_reports_skip_counts():
    """I4: evaluate() must expose how many users were skipped (missing truth /
    cold-start) so HR@10 vs HR@10_all divergence is diagnosable."""
    movies = pd.DataFrame(
        [
            {"movieId": 1, "title": "A (2000)", "genres": "Drama"},
            {"movieId": 2, "title": "B (2001)", "genres": "Drama"},
        ]
    )
    rows = []
    # user 1: 2 ratings -> train 1, test 1 (evaluable)
    rows += [
        {"userId": 1, "movieId": 1, "rating": 5.0, "timestamp": 100},
        {"userId": 1, "movieId": 2, "rating": 4.0, "timestamp": 200},
    ]
    # user 2: single rating -> test-only -> cold-start (skipped by evaluate)
    rows.append({"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": 300})
    ratings = pd.DataFrame(rows)

    cf, test, truth_by_user = prepare_eval(ratings)
    eligible = test["userId"].drop_duplicates()
    summary = evaluate(cf, movies, truth_by_user, eligible, top_k=2)

    assert int(summary.attrs["n_missing_truth"]) == 0
    # Both users end up cold-start in the recommend loop: user 2 has no train
    # ratings at all, and user 1's single train rating yields no CF candidates
    # on such a tiny graph (recommend_for_user raises ValueError).
    assert int(summary.attrs["n_cold_start"]) == 2
    assert int(summary.iloc[0]["users_evaluated"]) == 0
    # HR@10_all counts the skipped users in the denominator (0 hits / 2).
    assert summary.iloc[0]["HR@10_all"] == 0.0


def test_streamlit_app_parses():
    ast.parse((ROOT / "src" / "app" / "streamlit_app.py").read_text())


# --- fixtures for CF orphan test ---


@pytest.fixture
def cf_movies_mini() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "A", "genres": "Action"},
            {"movieId": 2, "title": "B", "genres": "Action"},
            {"movieId": 3, "title": "C", "genres": "Drama"},
            {"movieId": 4, "title": "D", "genres": "Drama"},
        ]
    )


@pytest.fixture
def cf_model_mini(cf_movies_mini) -> object:
    rows = []
    for uid in range(1, 12):
        rows += [
            {"userId": uid, "movieId": 1, "rating": 5.0, "timestamp": 1000 + uid},
            {"userId": uid, "movieId": 2, "rating": 4.5, "timestamp": 2000 + uid},
            {"userId": uid, "movieId": 3, "rating": 4.0, "timestamp": 3000 + uid},
        ]
    for uid in range(5, 12):
        rows.append(
            {"userId": uid, "movieId": 4, "rating": 4.5, "timestamp": 4000 + uid}
        )
    return build_cf_model(pd.DataFrame(rows))
