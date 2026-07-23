"""Streamlit app smoke tests — owned by QA (Kiên).

Cases A1-A2 cover Task T13/T20 deliverables. They are SKIPPED until
`data/processed/*.parquet` and Streamlit v1 are ready. Streamlit apps
are hard to unit-test cleanly, so these are intentionally lightweight:
they verify that imports succeed and the entry function doesn't blow up
on the missing-data branch.

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
sys.path.insert(0, str(ROOT))  # so `import src.app` works


def _require_processed():
    processed = ROOT / "data" / "processed" / "movies_clean.parquet"
    if not processed.exists():
        pytest.skip(
            "Parquet missing — Demo v1 (T13) cannot be tested yet. "
            "Run `python scripts/run_pipeline.py` first."
        )


def test_a1_missing_data_warns_cleanly(monkeypatch, tmp_path):
    """A1: when parquet is missing, app shows a warning — no traceback.

    Spec §11 risk: 'Title bị trùng hoặc nhập sai' style edge cases.
    """
    # TODO Kiên: implement after T13
    # Streamlit testing approach:
    #   from streamlit.testing.v1 import AppTest
    #   at = AppTest.from_file("src/app.py", default_timeout=10).run()
    #   assert not at.exception
    #   assert any("parquet" in w.lower() for w in at.warning)
    pytest.skip("TODO Kiên: implement after T13 Demo v1")


def test_a2_three_tabs_render():
    """A2: app exposes 3 tabs — Simple / Content / CF.

    Spec §7: UI 3 tab chính.
    """
    _require_processed()
    # TODO Kiên: implement after T13
    # from streamlit.testing.v1 import AppTest
    # at = AppTest.from_file("src/app.py", default_timeout=10).run()
    # tab_titles = [t.label for t in at.tabs]
    # assert "Simple Recommender" in tab_titles
    # assert "Content-based" in tab_titles
    # assert "Collaborative Filtering" in tab_titles
    pytest.skip("TODO Kiên: implement after T13 Demo v1")
