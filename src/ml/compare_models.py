from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

EVALUATION_DIR = BASE_DIR / "evaluation"

POPULAR_PATH = (
    EVALUATION_DIR
    / "popular"
    / "popular_summary.json"
)

MODEL_PATH = (
    EVALUATION_DIR
    / "evaluation_summary.json"
)

HYBRID_PATH = (
    EVALUATION_DIR
    / "hybrid"
    / "hybrid_summary.json"
)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    popular_summary = load_json(
        POPULAR_PATH
    )

    model_summaries = load_json(
        MODEL_PATH
    )

    hybrid_summaries = load_json(
        HYBRID_PATH
    )

    # Lấy alpha tốt nhất
    best_hybrid = max(
        hybrid_summaries,
        key=lambda item: (
            item.get("precision@10", 0),
            item.get("recall@10", 0),
        ),
    )

    all_results = [
        popular_summary,
        *model_summaries,
        best_hybrid,
    ]

    result = pd.DataFrame(
        all_results
    )

    columns = [
        "model",
        "alpha",
        "top_k",
        "positive_threshold",
        "evaluated_users",
        "precision@10",
        "recall@10",
        "hit_rate@10",
        "average_matches",
        "average_truth_items",
    ]

    columns = [
        column
        for column in columns
        if column in result.columns
    ]

    result = result[columns]

    print("\n" + "=" * 75)
    print("SO SÁNH TẤT CẢ MÔ HÌNH")
    print("=" * 75)

    print(
        result.to_string(
            index=False
        )
    )

    result.to_csv(
        EVALUATION_DIR
        / "all_model_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()