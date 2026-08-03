from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
FIXTURE_DATA_DIR = ROOT / "tests" / "fixtures" / "data" / "processed"
FIXTURE_MODEL_DIR = ROOT / "tests" / "fixtures" / "model"


def _reload_app_modules(monkeypatch):
    monkeypatch.setenv("REC_DATA_DIR", str(FIXTURE_DATA_DIR))
    monkeypatch.setenv("REC_MODEL_DIR", str(FIXTURE_MODEL_DIR))

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    for module_name in ["app.streamlit_app", "app.model_adapter"]:
        sys.modules.pop(module_name, None)


def test_a1_model_adapter_predicts_from_fixture(monkeypatch):
    _reload_app_modules(monkeypatch)

    model_adapter = importlib.import_module("app.model_adapter")
    movies, ratings = model_adapter.load_data()
    recommendations = model_adapter.predict(
        user_id=1,
        movies=movies,
        ratings=ratings,
        top_k=3,
    )

    required_columns = {
        "rank",
        "movieId",
        "title",
        "genres",
        "rating",
        "num_ratings",
        "model_score",
    }
    user_seen = set(
        ratings.loc[ratings["userId"] == 1, "movieId"].astype(int)
    )

    assert required_columns <= set(recommendations.columns)
    assert len(recommendations) == 3
    assert recommendations["rank"].tolist() == [1, 2, 3]
    assert set(recommendations["movieId"].astype(int)).isdisjoint(user_seen)
    assert recommendations["model_score"].is_monotonic_decreasing


def test_a2_streamlit_app_imports_current_layout(monkeypatch):
    _reload_app_modules(monkeypatch)

    streamlit_app = importlib.import_module("app.streamlit_app")

    assert hasattr(streamlit_app, "render_result_rows")
    assert hasattr(streamlit_app, "render_context")
    assert hasattr(streamlit_app, "main")
