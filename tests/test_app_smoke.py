"""Streamlit app smoke tests — owned by QA (Kiên).

Cases A1-A2 cover Task T09/T13/T20 deliverables. Lightweight checks:
imports succeed and the 3-tab entry path handles missing data cleanly.

For full UX testing, do manual smoke testing in the browser:

    streamlit run src/app.py

Reference: src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def test_a1_missing_data_warns_cleanly(monkeypatch):
    """A1: when parquet is missing, app shows a warning — no traceback."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    # Point ROOT's processed path at an empty temp tree via monkeypatch on
    # the Path.exists check used inside app.main. Easiest reliable approach:
    # run the app after temporarily renaming the expected parquet via env-
    # independent monkeypatch of Path.exists for that specific file.
    processed = ROOT / "data" / "processed" / "movies_clean.parquet"
    original_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.resolve() == processed.resolve():
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=15).run()
    assert not at.exception
    warning_text = " ".join(w.value for w in at.warning).lower()
    assert "parquet" in warning_text


def test_a2_three_tabs_render():
    """A2: app exposes 3 tabs — Simple / Content / CF."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    processed = ROOT / "data" / "processed" / "movies_clean.parquet"
    fixture = ROOT / "tests" / "fixtures" / "data" / "processed" / "movies_clean.parquet"
    if not processed.exists() and not fixture.exists():
        pytest.skip(
            "Parquet missing — run `python scripts/run_pipeline.py` "
            "or `python tests/fixtures/build_test_assets.py` first."
        )

    # App hard-codes data/processed; skip A2 when only fixtures exist.
    if not processed.exists():
        pytest.skip("Full processed parquet required for A2 tab render smoke.")

    at = AppTest.from_file(str(ROOT / "src" / "app.py"), default_timeout=60).run()
    assert not at.exception
    tab_labels = [t.label for t in at.tabs]
    assert "Simple Recommender" in tab_labels
    assert "Content-based" in tab_labels
    assert "Collaborative Filtering" in tab_labels
