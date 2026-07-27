from pathlib import Path
from typing import Callable
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from recommend import (
    recommend_for_user_cf,
    recommend_for_user_content,
)


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# File nằm tại:
# hybrid-movie-recommender/src/ml/evaluate_models.py

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

CF_MODEL_DIR = BASE_DIR / "model" / "knn_cf"

CONTENT_MODEL_DIR = (
    BASE_DIR / "model" / "knn_content"
)

EVALUATION_DIR = BASE_DIR / "evaluation"

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
    """
    Kiểm tra file có tồn tại hay không.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file:\n{file_path}"
        )


# =========================================================
# 3. LOAD TRAIN/TEST
# =========================================================

def load_evaluation_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Đọc dữ liệu train và test.
    """

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

    train_missing = (
        required_columns
        - set(train_ratings.columns)
    )

    test_missing = (
        required_columns
        - set(test_ratings.columns)
    )

    if train_missing:
        raise ValueError(
            f"Train thiếu cột: {train_missing}"
        )

    if test_missing:
        raise ValueError(
            f"Test thiếu cột: {test_missing}"
        )

    return train_ratings, test_ratings


# =========================================================
# 4. LOAD MODEL CF
# =========================================================

def load_cf_artifacts():
    """
    Load model, matrix và mappings của CF.
    """

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
# 5. LOAD MODEL CONTENT
# =========================================================

def load_content_artifacts():
    """
    Load matrix và mappings của Content-Based.

    KNN Content model không bắt buộc dùng trong evaluation
    vì hàm đánh giá tạo user profile và tính cosine trực tiếp.
    """

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
    """
    Chuẩn hóa mapping về hai dictionary:

    movieId -> matrix index
    matrix index -> movieId
    """

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
        movie_ids = mappings["movie_ids"]

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
        "Không tìm thấy movie mapping "
        "trong file mappings."
    )


# =========================================================
# 7. CÁC CHỈ SỐ ĐÁNH GIÁ
# =========================================================

def precision_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    Precision@K:
    Số phim dự đoán đúng / K.
    """

    if k <= 0:
        raise ValueError(
            "k phải lớn hơn 0."
        )

    predicted_at_k = predicted[:k]

    if not predicted_at_k:
        return 0.0

    matched = (
        set(predicted_at_k)
        & truth
    )

    return len(matched) / k


def recall_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    Recall@K:
    Số phim dự đoán đúng /
    tổng số phim positive trong test.
    """

    if not truth:
        return 0.0

    predicted_at_k = set(
        predicted[:k]
    )

    matched = (
        predicted_at_k
        & truth
    )

    return len(matched) / len(truth)


def hit_rate_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    HitRate@K:
    Bằng 1 nếu có ít nhất một phim dự đoán đúng.
    """

    predicted_at_k = set(
        predicted[:k]
    )

    matched = (
        predicted_at_k
        & truth
    )

    return float(bool(matched))


# =========================================================
# 8. CHỌN USER ĐÁNH GIÁ
# =========================================================

def select_evaluation_users(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    sample_size: int | None = 100,
    random_state: int = 42,
    positive_threshold: float = 4.0,
) -> list[int]:
    """
    Chọn các user:
    - Có lịch sử trong train.
    - Có ít nhất một rating positive trong test.
    """

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

    if (
        sample_size is not None
        and len(eligible_users) > sample_size
    ):
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
# 9. ĐÁNH GIÁ MỘT MODEL
# =========================================================

def evaluate_model(
    model_name: str,
    test_ratings: pd.DataFrame,
    recommend_function: Callable[
        [int],
        list[int],
    ],
    user_ids: list[int],
    model_catalog: set[int],
    top_k: int = 10,
    positive_threshold: float = 4.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Đánh giá một model trên danh sách user.

    recommend_function nhận user_id
    và trả về list movieId.
    """

    results = []

    total_users = len(user_ids)

    for position, user_id in enumerate(
        user_ids,
        start=1,
    ):
        user_test = test_ratings.loc[
            test_ratings["userId"] == user_id
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

        # Chỉ giữ các phim model có trong catalog
        truth = truth & model_catalog

        if not truth:
            continue

        predicted = recommend_function(
            int(user_id)
        )

        matched = (
            set(predicted[:top_k])
            & truth
        )

        results.append(
            {
                "model": model_name,
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
                        predicted=predicted,
                        truth=truth,
                        k=top_k,
                    )
                ),
                f"recall@{top_k}": (
                    recall_at_k(
                        predicted=predicted,
                        truth=truth,
                        k=top_k,
                    )
                ),
                f"hit_rate@{top_k}": (
                    hit_rate_at_k(
                        predicted=predicted,
                        truth=truth,
                        k=top_k,
                    )
                ),
            }
        )

        if position % 20 == 0:
            print(
                f"{model_name}: "
                f"đã xử lý "
                f"{position}/{total_users} user"
            )

    user_results = pd.DataFrame(results)

    if user_results.empty:
        summary = {
            "model": model_name,
            "top_k": int(top_k),
            "positive_threshold": float(
                positive_threshold
            ),
            "evaluated_users": 0,
            f"precision@{top_k}": 0.0,
            f"recall@{top_k}": 0.0,
            f"hit_rate@{top_k}": 0.0,
            "average_matches": 0.0,
            "average_truth_items": 0.0,
        }

        return summary, user_results

    summary = {
        "model": model_name,
        "top_k": int(top_k),
        "positive_threshold": float(
            positive_threshold
        ),
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
    """
    Đánh giá KNN-CF và Content-Based
    trên cùng một tập user.
    """

    top_k = 10

    positive_threshold = 4.0

    sample_size = 100

    neighbors_per_movie = 30

    print("=" * 65)
    print("EVALUATE CF AND CONTENT-BASED MODELS")
    print("=" * 65)

    # -----------------------------------------------------
    # Load dữ liệu
    # -----------------------------------------------------

    print("\n[1/5] Loading evaluation data...")

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

    # -----------------------------------------------------
    # Chọn user
    # -----------------------------------------------------

    print("\n[2/5] Selecting users...")

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
        f"Số user được chọn: "
        f"{len(evaluation_users):,}"
    )

    # -----------------------------------------------------
    # Load CF
    # -----------------------------------------------------

    print("\n[3/5] Evaluating CF model...")

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

    def cf_recommender(
        user_id: int,
    ) -> list[int]:
        return recommend_for_user_cf(
            user_id=user_id,
            train_ratings=train_ratings,
            model=cf_model,
            movie_user_matrix=cf_matrix,
            movie_id_to_index=(
                cf_movie_id_to_index
            ),
            index_to_movie_id=(
                cf_index_to_movie_id
            ),
            top_k=top_k,
            neighbors_per_movie=(
                neighbors_per_movie
            ),
            positive_threshold=(
                positive_threshold
            ),
        )

    (
        cf_summary,
        cf_user_results,
    ) = evaluate_model(
        model_name="KNN_CF",
        test_ratings=test_ratings,
        recommend_function=cf_recommender,
        user_ids=evaluation_users,
        model_catalog=set(
            cf_movie_id_to_index.keys()
        ),
        top_k=top_k,
        positive_threshold=(
            positive_threshold
        ),
    )

    # -----------------------------------------------------
    # Load Content-Based
    # -----------------------------------------------------

    print(
        "\n[4/5] Evaluating "
        "Content-Based model..."
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

    def content_recommender(
        user_id: int,
    ) -> list[int]:
        return recommend_for_user_content(
            user_id=user_id,
            train_ratings=train_ratings,
            movie_feature_matrix=(
                content_matrix
            ),
            movie_id_to_index=(
                content_movie_id_to_index
            ),
            index_to_movie_id=(
                content_index_to_movie_id
            ),
            top_k=top_k,
            positive_threshold=(
                positive_threshold
            ),
        )

    (
        content_summary,
        content_user_results,
    ) = evaluate_model(
        model_name="KNN_CONTENT",
        test_ratings=test_ratings,
        recommend_function=(
            content_recommender
        ),
        user_ids=evaluation_users,
        model_catalog=set(
            content_movie_id_to_index.keys()
        ),
        top_k=top_k,
        positive_threshold=(
            positive_threshold
        ),
    )

    # -----------------------------------------------------
    # Lưu kết quả
    # -----------------------------------------------------

    print(
        "\n[5/5] Saving evaluation results..."
    )

    summaries = [
        cf_summary,
        content_summary,
    ]

    summary_dataframe = pd.DataFrame(
        summaries
    )

    print("\n" + "=" * 65)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 65)

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    summary_dataframe.to_csv(
        EVALUATION_DIR
        / "model_comparison.csv",
        index=False,
    )

    cf_user_results.to_csv(
        EVALUATION_DIR
        / "cf_user_metrics.csv",
        index=False,
    )

    content_user_results.to_csv(
        EVALUATION_DIR
        / "content_user_metrics.csv",
        index=False,
    )

    with open(
        EVALUATION_DIR
        / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summaries,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print(
        f"\nKết quả được lưu tại:\n"
        f"{EVALUATION_DIR}"
    )


if __name__ == "__main__":
    main()