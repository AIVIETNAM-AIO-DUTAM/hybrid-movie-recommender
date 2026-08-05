from pathlib import Path
import json

import numpy as np
import pandas as pd

from metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
)

from popular_baseline import (
    build_popular_ranking,
    recommend_for_user_popular,
)


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# File nằm tại:
# hybrid-movie-recommender/src/ml/evaluate_popular.py

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)

EVALUATION_DIR = (
    BASE_DIR
    / "evaluation"
    / "popular"
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_PATH = (
    DATA_DIR
    / "rating_cf_train.parquet"
)

TEST_PATH = (
    DATA_DIR
    / "rating_cf_test.parquet"
)

MOVIES_PATH = (
    DATA_DIR
    / "movies_clean.parquet"
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
# 3. LOAD DỮ LIỆU
# =========================================================

def load_evaluation_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Đọc tập train và test.
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
            f"Train thiếu các cột: {train_missing}"
        )

    if test_missing:
        raise ValueError(
            f"Test thiếu các cột: {test_missing}"
        )

    return train_ratings, test_ratings


# =========================================================
# 4. CHỌN USER ĐÁNH GIÁ
# =========================================================

def select_evaluation_users(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    sample_size: int | None = 100,
    random_state: int = 42,
    positive_threshold: float = 4.0,
) -> list[int]:
    """
    Chọn user:
    - Có lịch sử trong train.
    - Có ít nhất một phim positive trong test.
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
        train_users & positive_test_users
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
# 5. ĐÁNH GIÁ MOST POPULAR
# =========================================================

def evaluate_popular_model(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    popular_ranking: pd.DataFrame,
    user_ids: list[int],
    top_k: int = 10,
    positive_threshold: float = 4.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Đánh giá Most Popular bằng:
    - Precision@K
    - Recall@K
    - HitRate@K
    """

    results = []

    popular_catalog = set(
        popular_ranking["movieId"]
        .astype(int)
        .tolist()
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

        # Chỉ đánh giá các phim thuộc catalog train
        truth = truth & popular_catalog

        if not truth:
            continue

        predicted = recommend_for_user_popular(
            user_id=int(user_id),
            train_ratings=train_ratings,
            popular_ranking=popular_ranking,
            top_k=top_k,
        )

        matched = (
            set(predicted[:top_k])
            & truth
        )

        results.append(
            {
                "model": "MOST_POPULAR",
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
                f"Đã xử lý "
                f"{position}/{total_users} user"
            )

    user_results = pd.DataFrame(
        results
    )

    if user_results.empty:
        summary = {
            "model": "MOST_POPULAR",
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
        "model": "MOST_POPULAR",
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
# 6. LƯU TOP PHIM PHỔ BIẾN
# =========================================================

def save_popular_movies(
    popular_ranking: pd.DataFrame,
) -> None:
    """
    Ghép bảng Popular với thông tin tên phim nếu file movies tồn tại.
    """

    if not MOVIES_PATH.exists():
        popular_ranking.to_csv(
            EVALUATION_DIR
            / "popular_movie_ranking.csv",
            index=False,
        )

        return

    movies = pd.read_parquet(
        MOVIES_PATH
    )

    movie_columns = [
        column
        for column in [
            "movieId",
            "title",
            "genres",
        ]
        if column in movies.columns
    ]

    popular_with_information = (
        popular_ranking.merge(
            movies[movie_columns],
            on="movieId",
            how="left",
        )
    )

    output_columns = [
        column
        for column in [
            "movieId",
            "title",
            "genres",
            "positive_count",
            "rating_count",
            "average_rating",
        ]
        if column
        in popular_with_information.columns
    ]

    popular_with_information[
        output_columns
    ].to_csv(
        EVALUATION_DIR
        / "popular_movie_ranking.csv",
        index=False,
    )


# =========================================================
# 7. MAIN
# =========================================================

def main() -> None:
    top_k = 10

    positive_threshold = 4.0

    # Phải dùng cùng sample_size và random_state
    # như CF, Content và Hybrid để so sánh công bằng.
    sample_size = 100

    random_state = 42

    print("=" * 65)
    print("EVALUATE MOST POPULAR BASELINE")
    print("=" * 65)

    print("\n[1/4] Loading train/test data...")

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

    print("\n[2/4] Building Popular ranking...")

    popular_ranking = (
        build_popular_ranking(
            train_ratings=train_ratings,
            positive_threshold=(
                positive_threshold
            ),
        )
    )

    print(
        f"Số phim trong bảng xếp hạng: "
        f"{len(popular_ranking):,}"
    )

    print("\n[3/4] Selecting users...")

    evaluation_users = (
        select_evaluation_users(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            sample_size=sample_size,
            random_state=random_state,
            positive_threshold=(
                positive_threshold
            ),
        )
    )

    print(
        f"Số user được đánh giá: "
        f"{len(evaluation_users):,}"
    )

    print(
        "\n[4/4] Evaluating "
        "Most Popular baseline..."
    )

    summary, user_results = (
        evaluate_popular_model(
            train_ratings=train_ratings,
            test_ratings=test_ratings,
            popular_ranking=popular_ranking,
            user_ids=evaluation_users,
            top_k=top_k,
            positive_threshold=(
                positive_threshold
            ),
        )
    )

    print("\n" + "=" * 65)
    print("KẾT QUẢ MOST POPULAR BASELINE")
    print("=" * 65)

    summary_dataframe = pd.DataFrame(
        [summary]
    )

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    summary_dataframe.to_csv(
        EVALUATION_DIR
        / "popular_summary.csv",
        index=False,
    )

    user_results.to_csv(
        EVALUATION_DIR
        / "popular_user_metrics.csv",
        index=False,
    )

    save_popular_movies(
        popular_ranking
    )

    with open(
        EVALUATION_DIR
        / "popular_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
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