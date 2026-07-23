"""Streamlit demo — owned by Tech Lead (wires models from other roles)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommendation System")
st.caption("Simple · Content-based · Collaborative Filtering")

ARTIFACTS_DIR = ROOT / "artifacts"


def _cf_artifact_mtime() -> float:
    """Return mtime of item_similarity.npz, or 0.0 when missing.

    Used as part of the cache key so we auto-detect when Loan rebuilds
    artifacts without the user having to manually clear the Streamlit cache.
    """
    p = ARTIFACTS_DIR / "item_similarity.npz"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=True)
def load_data():
    from data_processing import load_processed

    return load_processed()


@st.cache_data(show_spinner=True, hash_funcs={Path: lambda _: 0})
def load_cf_cached(_artifacts_dir: Path, _mtime: float):
    """Load CF artifacts if present. Returns None when not built yet.

    Cache key includes the artifact file mtime: if Loan rebuilds
    `item_similarity.npz` after the first call, the next interaction picks
    up the new artifacts instead of returning the cached `None`. Pass the
    real mtime (from `_cf_artifact_mtime()`) at the call site.
    """
    if not (ARTIFACTS_DIR / "item_similarity.npz").exists():
        return None
    from recommender_cf import load_cf_artifacts

    return load_cf_artifacts()


@st.cache_data(show_spinner=True)
def build_content_cached(movies):
    from recommender_content import build_content_model

    return build_content_model(movies)


@st.cache_data(show_spinner=True)
def build_movie_stats_cached(ratings):
    """Cache movie stats so 25M-rating groupby runs once per session.

    Without this, every keystroke in the Simple tab or every CF fallback
    branch re-runs the groupby, making the demo sluggish.
    """
    from recommender_simple import build_movie_stats

    return build_movie_stats(ratings)


def _fallback_simple(movies, ratings, top_k: int, message: str) -> None:
    """Show `message` and render Simple Recommender as the cold-start fallback."""
    st.info(message)
    stats = build_movie_stats_cached(ratings)
    from recommender_simple import recommend_top_movies

    st.dataframe(
        recommend_top_movies(movies, stats, top_k=top_k),
        use_container_width=True,
    )


def main() -> None:
    tab_simple, tab_content, tab_cf = st.tabs(
        ["Simple Recommender", "Content-based", "Collaborative Filtering"]
    )

    processed = ROOT / "data" / "processed" / "movies_clean.parquet"
    if not processed.exists():
        st.warning(
            "Chưa có `data/processed/*.parquet`. "
            "Data Engineer chạy: `python scripts/run_pipeline.py` sau khi có CSV trong `data/raw/`."
        )
        st.stop()

    movies, ratings = load_data()

    with tab_simple:
        st.subheader("Top movies (weighted rating)")
        top_k = st.slider("top_k", 5, 50, 10, key="simple_k")
        genre = st.text_input("Filter genre (optional)", key="simple_genre")
        from recommender_simple import recommend_top_movies

        stats = build_movie_stats_cached(ratings)
        result = recommend_top_movies(
            movies, stats, top_k=top_k, genre=genre or None
        )
        st.dataframe(result, use_container_width=True)

    with tab_content:
        st.subheader("Similar movies by genre")
        query = st.text_input("movieId hoặc title", value="Toy Story (1995)")
        top_k = st.slider("top_k", 5, 50, 10, key="content_k")

        # Build once per session (cached). Cheap on 62k rows but still wasteful
        # to redo per click.
        model = build_content_cached(movies)
        from recommender_content import recommend_similar_movies

        if st.button("Recommend similar", key="content_btn"):
            try:
                movie_key: int | str = int(query) if query.strip().isdigit() else query
                result = recommend_similar_movies(model, movie_key, top_k=top_k)
                st.dataframe(result, use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))

    with tab_cf:
        st.subheader("Personalized for userId")
        user_id = st.number_input("userId", min_value=1, value=1, step=1)
        top_k = st.slider("top_k", 5, 50, 10, key="cf_k")
        from recommender_cf import recommend_for_user

        # Lazy-load artifacts. We pass the file mtime as part of the cache
        # key so a freshly built artifact is picked up without manual cache clear.
        cf = load_cf_cached(ARTIFACTS_DIR, _cf_artifact_mtime())

        if cf is None:
            _fallback_simple(
                movies, ratings, top_k,
                "Chưa build CF artifacts. Loan chạy: `python scripts/build_cf_artifacts.py`. "
                "Hiện fallback Simple Recommender.",
            )
        elif st.button("Recommend for user", key="cf_btn"):
            try:
                result = recommend_for_user(cf, movies, int(user_id), top_k=top_k)
                if result.empty:
                    _fallback_simple(
                        movies, ratings, top_k,
                        "Không có candidate CF cho user này → fallback Simple.",
                    )
                else:
                    st.dataframe(result, use_container_width=True)
            except KeyError:
                _fallback_simple(
                    movies, ratings, top_k,
                    "User không có trong train set → fallback Simple Recommender.",
                )
            except ValueError as exc:
                # No-liked or no-candidates signals from recommend_for_user.
                _fallback_simple(
                    movies, ratings, top_k,
                    f"CF không có candidate ({exc}). Fallback Simple.",
                )
            except Exception as exc:  # noqa: BLE001 — clean UI, not traceback
                st.error(f"CF không khả dụng: {exc}. Hiện fallback Simple.")
                stats = build_movie_stats_cached(ratings)
                from recommender_simple import recommend_top_movies

                st.dataframe(
                    recommend_top_movies(movies, stats, top_k=top_k),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
