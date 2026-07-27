from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. GỢI Ý CHO USER BẰNG COLLABORATIVE FILTERING
# =========================================================

def recommend_for_user_cf(
    user_id: int,
    train_ratings: pd.DataFrame,
    model: NearestNeighbors,
    movie_user_matrix: csr_matrix,
    movie_id_to_index: dict,
    index_to_movie_id: dict,
    top_k: int = 10,
    neighbors_per_movie: int = 30,
    positive_threshold: float = 4.0,
) -> list[int]:
    """
    Gợi ý phim cho user bằng Item-Based KNN Collaborative Filtering.

    Quy trình:
    1. Lấy các phim user đã đánh giá trong train.
    2. Chọn các phim user thích.
    3. Tìm các phim hàng xóm của từng phim đã thích.
    4. Cộng điểm similarity có trọng số rating.
    5. Loại các phim user đã xem.
    6. Trả về Top-K phim.
    """

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId", "rating"],
    ].copy()

    if user_history.empty:
        return []

    seen_movie_ids = set(
        user_history["movieId"]
        .astype(int)
        .tolist()
    )

    liked_movies = user_history.loc[
        user_history["rating"] >= positive_threshold
    ].copy()

    # Nếu không có rating >= positive_threshold,
    # lấy 5 phim user đánh giá cao nhất
    if liked_movies.empty:
        liked_movies = (
            user_history
            .sort_values(
                by="rating",
                ascending=False,
            )
            .head(5)
            .copy()
        )

    candidate_scores = defaultdict(float)

    number_of_neighbors = min(
        neighbors_per_movie + 1,
        movie_user_matrix.shape[0],
    )

    for row in liked_movies.itertuples(index=False):
        movie_id = int(row.movieId)
        rating = float(row.rating)

        if movie_id not in movie_id_to_index:
            continue

        movie_index = movie_id_to_index[movie_id]

        distances, indices = model.kneighbors(
            movie_user_matrix[movie_index],
            n_neighbors=number_of_neighbors,
        )

        for neighbor_index, distance in zip(
            indices.flatten(),
            distances.flatten(),
        ):
            neighbor_index = int(neighbor_index)

            neighbor_movie_id = int(
                index_to_movie_id[neighbor_index]
            )

            if neighbor_movie_id == movie_id:
                continue

            if neighbor_movie_id in seen_movie_ids:
                continue

            similarity = 1.0 - float(distance)

            if similarity <= 0:
                continue

            # Rating 4.0 -> 0.8
            # Rating 5.0 -> 1.0
            rating_weight = rating / 5.0

            candidate_scores[neighbor_movie_id] += (
                similarity * rating_weight
            )

    ranked_movies = sorted(
        candidate_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    recommendations = [
        int(movie_id)
        for movie_id, _ in ranked_movies[:top_k]
    ]

    return recommendations


# =========================================================
# 2. TẠO USER PROFILE CHO CONTENT-BASED
# =========================================================

def build_content_user_profile(
    user_history: pd.DataFrame,
    movie_feature_matrix: csr_matrix,
    movie_id_to_index: dict,
    positive_threshold: float = 4.0,
) -> csr_matrix | None:
    """
    Tạo vector sở thích của user từ các vector TF-IDF của phim.

    Các phim có rating cao sẽ có trọng số lớn hơn.
    """

    liked_movies = user_history.loc[
        user_history["rating"] >= positive_threshold
    ].copy()

    if liked_movies.empty:
        liked_movies = (
            user_history
            .sort_values(
                by="rating",
                ascending=False,
            )
            .head(5)
            .copy()
        )

    movie_indices = []
    weights = []

    for row in liked_movies.itertuples(index=False):
        movie_id = int(row.movieId)
        rating = float(row.rating)

        if movie_id not in movie_id_to_index:
            continue

        movie_indices.append(
            movie_id_to_index[movie_id]
        )

        # Rating 4.0 -> trọng số 1.0
        # Rating 4.5 -> trọng số 1.5
        # Rating 5.0 -> trọng số 2.0
        weight = rating - (
            positive_threshold - 1.0
        )

        weights.append(
            max(weight, 0.1)
        )

    if not movie_indices:
        return None

    movie_indices_array = np.asarray(
        movie_indices,
        dtype=np.int64,
    )

    weights_array = np.asarray(
        weights,
        dtype=np.float32,
    )

    selected_features = (
        movie_feature_matrix[
            movie_indices_array
        ]
    )

    weighted_features = (
        selected_features.multiply(
            weights_array.reshape(-1, 1)
        )
    )

    profile = (
        weighted_features.sum(axis=0)
        / weights_array.sum()
    )

    return csr_matrix(profile)


# =========================================================
# 3. GỢI Ý CHO USER BẰNG CONTENT-BASED
# =========================================================

def recommend_for_user_content(
    user_id: int,
    train_ratings: pd.DataFrame,
    movie_feature_matrix: csr_matrix,
    movie_id_to_index: dict,
    index_to_movie_id: dict,
    top_k: int = 10,
    positive_threshold: float = 4.0,
) -> list[int]:
    """
    Gợi ý phim cho user bằng Content-Based User Profile.

    Quy trình:
    1. Lấy lịch sử phim user đã xem.
    2. Tạo vector sở thích từ các vector TF-IDF.
    3. So sánh vector user với toàn bộ phim.
    4. Loại các phim user đã xem.
    5. Trả về Top-K phim.
    """

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId", "rating"],
    ].copy()

    if user_history.empty:
        return []

    seen_movie_ids = set(
        user_history["movieId"]
        .astype(int)
        .tolist()
    )

    user_profile = build_content_user_profile(
        user_history=user_history,
        movie_feature_matrix=movie_feature_matrix,
        movie_id_to_index=movie_id_to_index,
        positive_threshold=positive_threshold,
    )

    if user_profile is None:
        return []

    similarities = cosine_similarity(
        user_profile,
        movie_feature_matrix,
    ).flatten()

    # Loại các phim user đã xem
    for seen_movie_id in seen_movie_ids:
        movie_index = movie_id_to_index.get(
            int(seen_movie_id)
        )

        if movie_index is not None:
            similarities[movie_index] = -np.inf

    valid_count = int(
        np.isfinite(similarities).sum()
    )

    if valid_count == 0:
        return []

    number_to_select = min(
        top_k,
        valid_count,
    )

    # Lấy nhanh các phần tử có điểm lớn nhất
    candidate_indices = np.argpartition(
        similarities,
        -number_to_select,
    )[-number_to_select:]

    # Sắp xếp các ứng viên theo similarity giảm dần
    ranked_indices = candidate_indices[
        np.argsort(
            similarities[candidate_indices]
        )[::-1]
    ]

    recommendations = []

    for movie_index in ranked_indices:
        movie_index = int(movie_index)

        if not np.isfinite(
            similarities[movie_index]
        ):
            continue

        movie_id = int(
            index_to_movie_id[movie_index]
        )

        recommendations.append(movie_id)

        if len(recommendations) >= top_k:
            break

    return recommendations