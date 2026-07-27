from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from hybrid_rcm import (
    recommend_for_user_hybrid,
)


# =========================================================
# 1. ĐƯỜNG DẪN
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

CF_MODEL_DIR = BASE_DIR / "model" / "knn_cf"

CONTENT_MODEL_DIR = (
    BASE_DIR / "model" / "knn_content"
)

EVALUATION_DIR = (
    BASE_DIR / "evaluation" / "hybrid"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_PATH = (
    DATA_DIR / "rating_cf_train.parquet"
)

TEST_PATH = (
    DATA_DIR / "rating_cf_test.parquet"
)


# =========================================================
# 2. KIỂM TRA FILE
# =========================================================

def check_file(
    file_path: Path,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{file_path}"
        )


# =========================================================
# 3. LOAD TRAIN VÀ TEST
# =========================================================

def load_evaluation_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    check_file(TRAIN_PATH)
    check_file(TEST_PATH)

    train_ratings = pd.read_parquet(
        TRAIN_PATH
    )

    test_ratings = pd.read_parquet(
        TEST_PATH
    )

    required_columns = {
        "userId",
        "movieId",
        "rating",
    }

    if not required_columns.issubset(
        train_ratings.columns
    ):
        raise ValueError(
            "Train ratings thiếu cột cần thiết."
        )

    if not required_columns.issubset(
        test_ratings.columns
    ):
        raise ValueError(
            "Test ratings thiếu cột cần thiết."
        )

    return train_ratings, test_ratings


# =========================================================
# 4. LOAD CF ARTIFACTS
# =========================================================

def load_cf_artifacts():
    model_path = (
        CF_MODEL_DIR
        / "knn_cf_model.joblib"
    )

    matrix_path = (
        CF_MODEL_DIR
        / "movie_user_matrix.npz"
    )

    mappings_path = (
        CF_MODEL_DIR
        / "cf_mappings.joblib"
    )

    check_file(model_path)
    check_file(matrix_path)
    check_file(mappings_path)

    model = joblib.load(model_path)

    matrix = load_npz(matrix_path)

    mappings = joblib.load(
        mappings_path
    )

    return model, matrix, mappings


# =========================================================
# 5. LOAD CONTENT ARTIFACTS
# =========================================================

def load_content_artifacts():
    matrix_path = (
        CONTENT_MODEL_DIR
        / "movie_feature_matrix.npz"
    )

    mappings_path = (
        CONTENT_MODEL_DIR
        / "content_mappings.joblib"
    )

    check_file(matrix_path)
    check_file(mappings_path)

    matrix = load_npz(matrix_path)

    mappings = joblib.load(
        mappings_path
    )

    return matrix, mappings


# =========================================================
# 6. CHUẨN HÓA MAPPING
# =========================================================

def normalize_mappings(
    mappings: dict,
) -> tuple[dict, dict]:
    if (
        "movie_id_to_index" in mappings
        and "index_to_movie_id" in mappings
    ):
        movie_id_to_index = {
            int(movie_id): int(index)
            for movie_id, index
            in mappings[
                "movie_id_to_index"
            ].items()
        }

        index_to_movie_id = {
            int(index): int(movie_id)
            for index, movie_id
            in mappings[
                "index_to_movie_id"
            ].items()
        }

        return (
            movie_id_to_index,
            index_to_movie_id,
        )

    if "movie_ids" in mappings:
        movie_ids = mappings[
            "movie_ids"
        ]

        movie_id_to_index = {
            int(movie_id): int(index)
            for index, movie_id
            in enumerate(movie_ids)
        }

        index_to_movie_id = {
            int(index): int(movie_id)
            for index, movie_id
            in enumerate(movie_ids)
        }

        return (
            movie_id_to_index,
            index_to_movie_id,
        )

    raise ValueError(
        "Không tìm thấy movie mapping."
    )


# =========================================================
# 7. METRICS
# =========================================================

def precision_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    if not predicted:
        return 0.0

    matched = (
        set(predicted[:k])
        & truth
    )

    return len(matched) / k


def recall_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    if not truth:
        return 0.0

    matched = (
        set(predicted[:k])
        & truth
    )

    return len(matched) / len(truth)


def hit_rate_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    matched = (
        set(predicted[:k])
        & truth
    )

    return float(bool(matched))


# =========================================================
# 8. CHỌN USER
# =========================================================

def select_evaluation_users(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    sample_size: int = 100,
    random_state: int = 42,
    positive_threshold: float = 4.0,
) -> list[int]:
    train_users = set(
        train_ratings["userId"]
        .astype(int)
        .unique()
    )

    positive_test_users = set(
        test_ratings.loc[
            test_ratings["rating"]
            >= positive_threshold,
            "userId",
        ]
        .astype(int)
        .unique()
    )

    eligible_users = sorted(
        train_users
        & positive_test_users
    )

    if len(eligible_users) > sample_size:
        random_generator = (
            np.random.default_rng(
                random_state
            )
        )

        eligible_users = (
            random_generator.choice(
                eligible_users,
                size=sample_size,
                replace=False,
            )
            .astype(int)
            .tolist()
        )

    return eligible_users


# =========================================================
# 9. ĐÁNH GIÁ HYBRID VỚI MỘT ALPHA
# =========================================================

def evaluate_hybrid_model(
    alpha: float,
    user_ids: list[int],
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,

    cf_model,
    cf_matrix,
    cf_movie_id_to_index: dict,
    cf_index_to_movie_id: dict,

    content_matrix,
    content_movie_id_to_index: dict,
    content_index_to_movie_id: dict,

    top_k: int = 10,
    positive_threshold: float = 4.0,
    neighbors_per_movie: int = 50,
    content_candidate_limit: int = 500,
) -> tuple[dict, pd.DataFrame]:
    results = []

    hybrid_catalog = (
        set(cf_movie_id_to_index.keys())
        | set(content_movie_id_to_index.keys())
    )

    total_users = len(user_ids)

    for position, user_id in enumerate(
        user_ids,
        start=1,
    ):
        user_test = test_ratings.loc[
            test_ratings["userId"]
            == user_id
        ]

        truth = set(
            user_test.loc[
                user_test["rating"]
                >= positive_threshold,
                "movieId",
            ]
            .astype(int)
            .tolist()
        )

        truth = truth & hybrid_catalog

        if not truth:
            continue

        predicted = (
            recommend_for_user_hybrid(
                user_id=int(user_id),
                train_ratings=train_ratings,

                cf_model=cf_model,
                cf_matrix=cf_matrix,
                cf_movie_id_to_index=(
                    cf_movie_id_to_index
                ),
                cf_index_to_movie_id=(
                    cf_index_to_movie_id
                ),

                content_matrix=content_matrix,
                content_movie_id_to_index=(
                    content_movie_id_to_index
                ),
                content_index_to_movie_id=(
                    content_index_to_movie_id
                ),

                top_k=top_k,
                alpha=alpha,
                neighbors_per_movie=(
                    neighbors_per_movie
                ),
                positive_threshold=(
                    positive_threshold
                ),
                content_candidate_limit=(
                    content_candidate_limit
                ),
            )
        )

        matched = (
            set(predicted[:top_k])
            & truth
        )

        results.append(
            {
                "model": "HYBRID",
                "alpha": float(alpha),
                "userId": int(user_id),
                "num_truth": int(
                    len(truth)
                ),
                "num_recommended": int(
                    len(predicted[:top_k])
                ),
                "num_matched": int(
                    len(matched)
                ),
                f"precision@{top_k}": (
                    precision_at_k(
                        predicted,
                        truth,
                        top_k,
                    )
                ),
                f"recall@{top_k}": (
                    recall_at_k(
                        predicted,
                        truth,
                        top_k,
                    )
                ),
                f"hit_rate@{top_k}": (
                    hit_rate_at_k(
                        predicted,
                        truth,
                        top_k,
                    )
                ),
            }
        )

        if position % 20 == 0:
            print(
                f"Alpha {alpha}: "
                f"{position}/{total_users} user"
            )

    user_results = pd.DataFrame(
        results
    )

    if user_results.empty:
        summary = {
            "model": "HYBRID",
            "alpha": float(alpha),
            "top_k": int(top_k),
            "evaluated_users": 0,
            f"precision@{top_k}": 0.0,
            f"recall@{top_k}": 0.0,
            f"hit_rate@{top_k}": 0.0,
            "average_matches": 0.0,
            "average_truth_items": 0.0,
        }

        return summary, user_results

    summary = {
        "model": "HYBRID",
        "alpha": float(alpha),
        "top_k": int(top_k),
        "evaluated_users": int(
            len(user_results)
        ),
        f"precision@{top_k}": float(
            user_results[
                f"precision@{top_k}"
            ].mean()
        ),
        f"recall@{top_k}": float(
            user_results[
                f"recall@{top_k}"
            ].mean()
        ),
        f"hit_rate@{top_k}": float(
            user_results[
                f"hit_rate@{top_k}"
            ].mean()
        ),
        "average_matches": float(
            user_results[
                "num_matched"
            ].mean()
        ),
        "average_truth_items": float(
            user_results[
                "num_truth"
            ].mean()
        ),
    }

    return summary, user_results


# =========================================================
# 10. MAIN
# =========================================================

def main() -> None:
    top_k = 10

    positive_threshold = 4.0

    sample_size = 100

    neighbors_per_movie = 50

    content_candidate_limit = 500

    alpha_values = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    print("=" * 70)
    print("EVALUATE HYBRID RECOMMENDER")
    print("=" * 70)

    print("\n[1/4] Loading data...")

    train_ratings, test_ratings = (
        load_evaluation_data()
    )

    print(
        f"Train rows: "
        f"{len(train_ratings):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_ratings):,}"
    )

    print("\n[2/4] Selecting users...")

    evaluation_users = (
        select_evaluation_users(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            sample_size=sample_size,
            random_state=42,
            positive_threshold=(
                positive_threshold
            ),
        )
    )

    print(
        f"Evaluation users: "
        f"{len(evaluation_users):,}"
    )

    print("\n[3/4] Loading models...")

    (
        cf_model,
        cf_matrix,
        cf_mappings,
    ) = load_cf_artifacts()

    (
        cf_movie_id_to_index,
        cf_index_to_movie_id,
    ) = normalize_mappings(
        cf_mappings
    )

    (
        content_matrix,
        content_mappings,
    ) = load_content_artifacts()

    (
        content_movie_id_to_index,
        content_index_to_movie_id,
    ) = normalize_mappings(
        content_mappings
    )

    print("\n[4/4] Evaluating alpha values...")

    all_summaries = []

    for alpha in alpha_values:
        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Đang đánh giá alpha = {alpha}"
        )

        summary, user_results = (
            evaluate_hybrid_model(
                alpha=alpha,
                user_ids=evaluation_users,
                train_ratings=train_ratings,
                test_ratings=test_ratings,

                cf_model=cf_model,
                cf_matrix=cf_matrix,
                cf_movie_id_to_index=(
                    cf_movie_id_to_index
                ),
                cf_index_to_movie_id=(
                    cf_index_to_movie_id
                ),

                content_matrix=content_matrix,
                content_movie_id_to_index=(
                    content_movie_id_to_index
                ),
                content_index_to_movie_id=(
                    content_index_to_movie_id
                ),

                top_k=top_k,
                positive_threshold=(
                    positive_threshold
                ),
                neighbors_per_movie=(
                    neighbors_per_movie
                ),
                content_candidate_limit=(
                    content_candidate_limit
                ),
            )
        )

        all_summaries.append(summary)

        user_results.to_csv(
            EVALUATION_DIR
            / f"hybrid_alpha_{alpha}_users.csv",
            index=False,
        )

    summary_dataframe = pd.DataFrame(
        all_summaries
    )

    summary_dataframe = (
        summary_dataframe
        .sort_values(
            by=[
                f"precision@{top_k}",
                f"hit_rate@{top_k}",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ HYBRID")
    print("=" * 70)

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    summary_dataframe.to_csv(
        EVALUATION_DIR
        / "hybrid_alpha_comparison.csv",
        index=False,
    )

    with open(
        EVALUATION_DIR
        / "hybrid_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            all_summaries,
            file,
            ensure_ascii=False,
            indent=4,
        )

    best_result = (
        summary_dataframe.iloc[0]
    )

    print("\nAlpha tốt nhất:")

    print(
        f"alpha = "
        f"{best_result['alpha']}"
    )

    print(
        f"precision@{top_k} = "
        f"{best_result[f'precision@{top_k}']:.6f}"
    )

    print(
        f"recall@{top_k} = "
        f"{best_result[f'recall@{top_k}']:.6f}"
    )

    print(
        f"hit_rate@{top_k} = "
        f"{best_result[f'hit_rate@{top_k}']:.6f}"
    )

    print(
        f"\nKết quả được lưu tại:\n"
        f"{EVALUATION_DIR}"
    )


if __name__ == "__main__":
    main()