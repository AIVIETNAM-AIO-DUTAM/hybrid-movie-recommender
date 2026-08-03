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
PROCESSED_MOVIES = ROOT / "data" / "processed" / "movies_clean.parquet"
CF_ARTIFACT_FILES = (
    "utility_matrix.npz",
    "item_similarity.npz",
    "movie_id_maps.pkl",
    "cf_build_meta.json",
)


def _cf_artifacts_ready(artifacts_dir: Path | None = None) -> bool:
    """True only when all three CF artifact files exist."""
    prefix = artifacts_dir or ARTIFACTS_DIR
    return all((prefix / name).exists() for name in CF_ARTIFACT_FILES)


def _cf_artifact_mtime() -> float:
    """Return mtime of item_similarity.npz, or 0.0 when missing.

    Used as part of the cache key so we auto-detect when Loan rebuilds
    artifacts without the user having to manually clear the Streamlit cache.
    """
    p = ARTIFACTS_DIR / "item_similarity.npz"
    return p.stat().st_mtime if p.exists() else 0.0


def _movies_parquet_mtime() -> float:
    """mtime of movies_clean.parquet — busts content-model cache on rebuild."""
    return PROCESSED_MOVIES.stat().st_mtime if PROCESSED_MOVIES.exists() else 0.0


@st.cache_data(show_spinner=True)
def load_data():
    """Load movies + CF ratings (ignore content-split ratings for the 3-tab app)."""
    from data_processing import load_processed

    movies, ratings_cf, _ratings_content = load_processed()
    return movies, ratings_cf


@st.cache_data(show_spinner=True)
def load_cf_cached(_artifacts_dir: Path, mtime: float, n_ratings: int = -1):
    """Load CF artifacts if present. Returns None when not built yet.

    Cache key includes `mtime` + `n_ratings` (non-underscore so Streamlit
    actually hashes them): if Loan rebuilds `item_similarity.npz` after the
    first call, the next interaction picks up the new artifacts instead of
    returning the cached `None`. Pass the real mtime (from
    `_cf_artifact_mtime()`) at the call site.

    `n_ratings` (len of current ratings_cf) is checked against cf_build_meta
    so a rebuilt parquet with stale artifacts falls back to Simple.

    Partial/corrupt artifact sets return None (Simple fallback) instead of
    crashing the Streamlit page.
    """
    if not _cf_artifacts_ready(ARTIFACTS_DIR):
        return None
    try:
        from recommender_cf import load_cf_artifacts

        expected = n_ratings if n_ratings >= 0 else None
        return load_cf_artifacts(expected_n_ratings=expected)
    except (FileNotFoundError, OSError, ValueError):
        return None


@st.cache_data(show_spinner=True)
def build_content_cached(movies, _movies_mtime: float = 0.0):
    """Build content model; `_movies_mtime` invalidates cache when parquet changes."""
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
            "Data Engineer chạy: `python -c 'from src.data_processing import run_pipeline; run_pipeline()'` sau khi có CSV trong `data/raw/`."
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
        if result.empty:
            st.info(
                "Không có phim khớp bộ lọc genre. Thử genre khác hoặc để trống."
            )
        else:
            st.dataframe(result, use_container_width=True)

    with tab_content:
        st.subheader("Similar movies by genre")
        query = st.text_input("movieId hoặc title", value="Toy Story (1995)")
        top_k = st.slider("top_k", 5, 50, 10, key="content_k")

        # Build once per session (cached). mtime busts cache when parquet rebuilds.
        try:
            model = build_content_cached(movies, _movies_parquet_mtime())
        except ValueError as exc:
            st.error(str(exc))
            model = None
        from recommender_content import recommend_similar_movies

        if st.button("Recommend similar", key="content_btn"):
            if model is None:
                st.error("Content model chưa sẵn sàng.")
            else:
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
        cf = load_cf_cached(ARTIFACTS_DIR, _cf_artifact_mtime(), len(ratings))

        if cf is None:
            _fallback_simple(
                movies, ratings, top_k,
                "Chưa build CF artifacts (thiếu hoặc lỗi file). Loan chạy: "
                "`python scripts/build_hybrid_artifacts.py`. Hiện fallback Simple Recommender.",
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
