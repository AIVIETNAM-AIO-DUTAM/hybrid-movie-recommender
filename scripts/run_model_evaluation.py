"""Reproduce model_comparison.csv + evaluation_summary.json (KNN_CF / KNN_CONTENT).

NOTE — different protocol from evaluation/cf_eval_scores.csv:

- `evaluation/cf_eval_scores.csv`   <- scripts/run_cf_evaluation.py
    item-based CF, LEAVE-LAST-OUT (single held-out rating per user).
    HR@10 = 0.01 on sample 200.  This is the metric cited in final_report §5.1.

- `evaluation/model_comparison.csv` + `evaluation_summary.json`  <- THIS SCRIPT
    KNN CF / KNN CONTENT, multi-item test set per user (avg_truth_items ~ 13),
    hit_rate@10 = 0.46 (KNN_CF).  Different split, different truth granularity.

Do NOT merge the two in the report — they measure different things and are
not apples-to-apples (see reports/cf_evaluation.md "Reproduce" section).

Usage
-----
    source .venv/bin/activate
    python scripts/run_model_evaluation.py

Requires trained artifacts under model/knn_cf/ and model/knn_content/
(built by scripts/build_hybrid_artifacts.py) plus rating_cf_train/test
parquets split by src/ml/train_knn_*.py.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    eval_script = ROOT / "src" / "ml" / "evaluate_model.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing {eval_script}")

    print("Đang chạy KNN model evaluation (multi-item truth protocol)...")
    subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=str(ROOT),
        check=True,
    )
    print(
        "Hoàn tất! model_comparison.csv + evaluation_summary.json đã được "
        "ghi vào evaluation/ (protocol: multi-item truth, KHÁC cf_eval_scores.csv)."
    )


if __name__ == "__main__":
    main()
