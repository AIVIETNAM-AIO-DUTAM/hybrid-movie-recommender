import pandas as pd


def build_popular_ranking(
    train_ratings: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    Xây dựng bảng xếp hạng phim phổ biến từ tập train.

    Thứ tự ưu tiên:
    1. Số lượt rating tích cực.
    2. Rating trung bình.
    3. Tổng số lượt rating.

    Chỉ sử dụng tập train để tránh data leakage.
    """

    required_columns = {
        "userId",
        "movieId",
        "rating",
    }

    missing_columns = (
        required_columns - set(train_ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột: {missing_columns}"
        )

    if not 0 <= positive_threshold <= 5:
        raise ValueError(
            "positive_threshold phải nằm trong khoảng 0 đến 5."
        )

    rating_statistics = (
        train_ratings
        .groupby("movieId")
        .agg(
            rating_count=("rating", "size"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    positive_statistics = (
        train_ratings.loc[
            train_ratings["rating"]
            >= positive_threshold
        ]
        .groupby("movieId")
        .size()
        .rename("positive_count")
        .reset_index()
    )

    popularity = rating_statistics.merge(
        positive_statistics,
        on="movieId",
        how="left",
    )

    popularity["positive_count"] = (
        popularity["positive_count"]
        .fillna(0)
        .astype(int)
    )

    popularity["movieId"] = (
        popularity["movieId"]
        .astype(int)
    )

    popularity["average_rating"] = (
        popularity["average_rating"]
        .astype(float)
    )

    popularity = (
        popularity
        .sort_values(
            by=[
                "positive_count",
                "average_rating",
                "rating_count",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return popularity


def recommend_for_user_popular(
    user_id: int,
    train_ratings: pd.DataFrame,
    popular_ranking: pd.DataFrame,
    top_k: int = 10,
) -> list[int]:
    """
    Gợi ý các phim phổ biến nhất mà user chưa xem.

    Most Popular là baseline không cá nhân hóa.
    Tất cả user dùng chung bảng xếp hạng phim,
    nhưng các phim user đã xem sẽ bị loại.
    """

    if top_k <= 0:
        raise ValueError(
            "top_k phải lớn hơn 0."
        )

    required_columns = {
        "userId",
        "movieId",
    }

    missing_columns = (
        required_columns - set(train_ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột trong train: {missing_columns}"
        )

    if "movieId" not in popular_ranking.columns:
        raise ValueError(
            "popular_ranking thiếu cột movieId."
        )

    seen_movie_ids = set(
        train_ratings.loc[
            train_ratings["userId"] == user_id,
            "movieId",
        ]
        .astype(int)
        .tolist()
    )

    recommendations = []

    for movie_id in popular_ranking["movieId"]:
        movie_id = int(movie_id)

        if movie_id in seen_movie_ids:
            continue

        recommendations.append(movie_id)

        if len(recommendations) >= top_k:
            break

    return recommendations