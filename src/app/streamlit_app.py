from __future__ import annotations

import pickle
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from app import model_adapter


st.set_page_config(
    page_title="MicroLens Workbench",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    [data-testid="stSidebar"] {
        background: #f4f6fb;
        border-right: 1px solid #e8ebf2;
    }
    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .brand {
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0;
        margin-bottom: 0;
    }
    .brand-sub {
        color: #8a94a6;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: .08rem;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.45rem;
        line-height: 1.1;
        font-weight: 850;
        margin: 0 0 .4rem;
        color: #202533;
    }
    .muted {
        color: #8b93a5;
        font-size: .92rem;
    }
    .panel-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #262b38;
        margin: .2rem 0 .85rem;
    }
    .metric-box {
        border: 1px solid #d9def0;
        border-radius: 8px;
        padding: .85rem .95rem;
        background: #eef0ff;
        margin: .55rem 0;
    }
    .metric-label {
        color: #7b6cff;
        font-size: .72rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .metric-value {
        color: #1f2633;
        font-size: 1rem;
        font-weight: 800;
        margin-top: .2rem;
    }
    .result-row {
        display: grid;
        grid-template-columns: 72px minmax(250px, 2.1fr) minmax(210px, 1.55fr) 90px 110px 112px;
        column-gap: 1rem;
        align-items: center;
        border: 1px solid #e0e4ee;
        border-radius: 8px;
        min-height: 72px;
        padding: .72rem 1rem;
        margin-bottom: .55rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(18, 24, 38, .03);
    }
    .table-head {
        color: #8b93a5;
        font-size: .74rem;
        font-weight: 800;
        text-transform: uppercase;
        background: #f8f9fc;
        box-shadow: none;
        min-height: 48px;
    }
    .header-cell {
        min-height: 48px;
        display: flex;
        align-items: center;
        color: #8b93a5;
        font-size: .78rem;
        font-weight: 850;
        text-transform: uppercase;
    }
    .header-cell-center {
        justify-content: center;
    }
    .header-cell-end {
        justify-content: flex-end;
    }
    .table-cell {
        min-width: 0;
        display: flex;
        align-items: center;
        height: 100%;
    }
    .numeric-cell {
        justify-content: center;
        color: #2a3040;
        font-size: 1.05rem;
        font-weight: 650;
    }
    .info-cell {
        justify-content: end;
    }
    .rank-pill {
        width: 2.25rem;
        height: 1.55rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 6px;
        background: #7b4dff;
        color: white;
        font-size: .82rem;
        font-weight: 850;
    }
    .truncate {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        min-width: 0;
    }
    .movie-title {
        color: #242a38;
        font-weight: 750;
    }
    .genre-text {
        color: #5d6678;
        font-size: .9rem;
    }
    .score-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 3.8rem;
        padding: .2rem .5rem;
        border-radius: 6px;
        background: #101827;
        color: white;
        font-size: .82rem;
        font-weight: 800;
    }
    .info-button-spacer {
        min-height: 72px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }
    .context-item {
        display: grid;
        grid-template-columns: 72px minmax(0, 1fr);
        column-gap: .75rem;
        align-items: center;
        min-height: 64px;
        padding: .6rem 0;
        border-bottom: 1px solid #edf0f5;
        overflow: hidden;
    }
    .poster-fake {
        width: 72px;
        height: 48px;
        border-radius: 6px;
        background: linear-gradient(135deg, #a78bfa, #22c55e);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 850;
        font-size: .8rem;
        line-height: 1;
        overflow: hidden;
        flex: 0 0 auto;
    }
    .context-body {
        min-width: 0;
        overflow: hidden;
    }
    .context-title {
        color: #303747;
        font-weight: 700;
        font-size: .86rem;
        line-height: 1.25;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .context-meta {
        color: #8b93a5;
        font-size: .78rem;
        line-height: 1.2;
        margin-top: .28rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        min-height: 2.45rem;
        font-weight: 750;
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: #15171d;
        border-color: #15171d;
    }
    @media (max-width: 900px) {
        .result-row {
            grid-template-columns: 48px minmax(0, 1fr) 88px;
            column-gap: .75rem;
        }
        .result-row > div:nth-child(3),
        .result-row > div:nth-child(4),
        .result-row > div:nth-child(5) {
            display: none;
        }
    }
</style>
"""


def ellipsize(value: object, limit: int = 70) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def render_movie_dialog(movie: pd.Series) -> None:
    title = str(movie.get("title", "Movie detail"))

    if hasattr(st, "dialog"):
        @st.dialog(title)
        def dialog() -> None:
            st.write(f"**movieId:** {int(movie['movieId'])}")
            st.write(f"**Genres:** {movie.get('genres', '')}")
            if "rating" in movie:
                st.write(f"**Rating:** {float(movie['rating']):.3f}")
            if "model_score" in movie:
                st.write(
                    f"**Model score:** {float(movie['model_score']):.3f}"
                )
            if "num_ratings" in movie:
                st.write(f"**Number of ratings:** {int(movie['num_ratings'])}")

        dialog()
    else:
        st.info(
            f"{title} | movieId={int(movie['movieId'])} | "
            f"genres={movie.get('genres', '')}"
        )


def render_result_rows(recommendations: pd.DataFrame) -> None:
    table = recommendations.copy()
    table["rank"] = table["rank"].map(lambda value: f"#{int(value)}")
    table["rating"] = table["rating"].map(lambda value: f"{float(value):.2f}")
    table["model_score"] = table["model_score"].map(
        lambda value: f"{float(value):.3f}"
    )

    st.dataframe(
        table[
            [
                "rank",
                "title",
                "genres",
                "rating",
                "model_score",
                "num_ratings",
            ]
        ],
        column_config={
            "rank": st.column_config.TextColumn(
                "Rank",
                width="small",
            ),
            "title": st.column_config.TextColumn(
                "Movie",
                width="large",
            ),
            "genres": st.column_config.TextColumn(
                "Genres",
                width="medium",
            ),
            "rating": st.column_config.TextColumn(
                "Rating",
                width="small",
            ),
            "model_score": st.column_config.TextColumn(
                "Score",
                width="small",
            ),
            "num_ratings": st.column_config.NumberColumn(
                "Ratings",
                width="small",
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="recommendation_table",
    )


def render_context(history: pd.DataFrame, user_id: int) -> None:
    st.markdown("<div class='panel-title'>📋 Context</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='muted'>User <b>{user_id}</b> · {len(history)} ratings</div>",
        unsafe_allow_html=True,
    )

    if history.empty:
        st.info("User này chưa có lịch sử rating trong dữ liệu.")
        return

    for _, row in history.head(12).iterrows():
        initials = str(row.get("title", "?"))[:2].upper()
        date_text = (
            row["rated_at"].strftime("%Y-%m-%d")
            if pd.notna(row.get("rated_at"))
            else "unknown date"
        )
        title = escape(str(row.get("title", "")))
        st.markdown(
            f"""
            <div class="context-item">
                <div class="poster-fake">{initials}</div>
                <div class="context-body">
                    <div class="context-title" title="{title}">
                        {escape(ellipsize(row.get('title', ''), 42))}
                    </div>
                    <div class="context-meta">
                        ⭐ {float(row['rating']):.1f} · {date_text}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        movies, ratings = model_adapter.load_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    users = model_adapter.get_user_options(ratings)

    with st.sidebar:
        st.markdown("<div class='brand'>🎬 MicroLens</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>Rec Workbench</div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("### Settings")

        model_name = st.radio(
            "Model Architecture",
            model_adapter.MODEL_OPTIONS,
            index=0,
            help="Hybrid gọi CF artifacts và Content artifacts, sau đó kết hợp score bằng alpha.",
        )

        selected_user = st.selectbox(
            "Chọn user id",
            users,
            index=0,
        )
        typed_user = st.text_input(
            "Hoặc nhập user id",
            placeholder="Bỏ trống để dùng user đã chọn",
        )
        if typed_user.strip():
            try:
                user_id = int(typed_user.strip())
            except ValueError:
                st.error("User id phải là số nguyên.")
                user_id = int(selected_user)
        else:
            user_id = int(selected_user)

        if model_adapter.SUPPORTS_TOP_K:
            top_k = st.slider(
                "Top-k results",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
            )
        else:
            top_k = 10

        predict_clicked = st.button(
            "🚀 Predict",
            type="primary",
            use_container_width=True,
        )
        clear_clicked = st.button(
            "× Clear results",
            use_container_width=True,
        )

        st.divider()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Active user</div>
                <div class="metric-value">👤 {user_id}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Model</div>
                <div class="metric-value">{model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if clear_clicked:
        st.session_state.pop("recommendations", None)
        st.session_state.pop("active_user_id", None)

    if predict_clicked:
        try:
            recs = model_adapter.predict(
                user_id=user_id,
                movies=movies,
                ratings=ratings,
                top_k=top_k,
                model_name=model_name,
            )
            st.session_state["active_user_id"] = user_id
            st.session_state["recommendations"] = recs
        except FileNotFoundError as exc:
            st.session_state.pop("recommendations", None)
            st.session_state.pop("active_user_id", None)
            st.error(f"Không tìm thấy file mô hình: {exc}")
        except ValueError as exc:
            st.session_state.pop("recommendations", None)
            st.session_state.pop("active_user_id", None)
            st.error(f"Lỗi dữ liệu đầu vào: {exc}")
        except (OSError, pickle.UnpicklingError) as exc:
            # Corrupt npz/joblib artifacts (truncated download, LFS pointer text, etc.)
            st.session_state.pop("recommendations", None)
            st.session_state.pop("active_user_id", None)
            st.error(
                "Mô hình bị hỏng hoặc chưa được pull đầy đủ qua Git LFS. "
                "Chạy `git lfs pull` rồi `python scripts/build_hybrid_artifacts.py`. "
                f"Chi tiết: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 — fallback UI phải không bao giờ crash trang
            st.session_state.pop("recommendations", None)
            st.session_state.pop("active_user_id", None)
            st.error(f"Lỗi không xác định khi sinh recommendation: {exc}")

    active_user_id = int(st.session_state.get("active_user_id", user_id))
    history = model_adapter.get_user_context(active_user_id, movies, ratings)

    center, right = st.columns([4.0, 1.35], gap="large")

    with center:
        st.markdown("<div class='hero-title'>⭐ Recommend</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='muted'>Generate recommendations based on user history.</div>",
            unsafe_allow_html=True,
        )
        st.caption("Nhập hoặc chọn user ở sidebar, chọn Top-k rồi nhấn Predict.")
        meta_cols = st.columns(3)
        meta_cols[0].metric("Model", model_name)
        meta_cols[1].metric("Top-k", top_k)
        meta_cols[2].metric("User", active_user_id)
        st.write("")

        recommendations = st.session_state.get("recommendations")
        if recommendations is None:
            st.info("Chọn user và nhấn Predict để hiện kết quả.")
        elif recommendations.empty:
            st.warning("Không tạo được recommendation cho user này.")
        else:
            st.caption(
                f"Showing {len(recommendations)} predictions via {model_name}"
            )
            render_result_rows(recommendations)

    with right:
        render_context(history, active_user_id)


if __name__ == "__main__":
    main()
