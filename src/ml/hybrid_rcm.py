from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


# =========================================================
# CHUẨN HÓA ĐIỂM VỀ KHOẢNG 0-1
# =========================================================

def normalize_scores(
    scores: dict[int, float],
) -> dict[int, float]:
    """
    Chuẩn hóa điểm bằng cách chia cho điểm lớn nhất.

    Điểm sau chuẩn hóa nằm trong khoảng 0-1.
    """

    if not scores:
        return {}

    max_score = max(scores.values())

    if max_score <= 0:
        return {
            int(movie_id): 0.0
            for movie_id in scores
        }

    return {
        int(movie_id): float(score / max_score)
        for movie_id, score in scores.items()
    }


# =========================================================
# TÍNH ĐIỂM COLLABORATIVE FILTERING
# =========================================================

def get_cf_candidate_scores(
    user_id: int,
    train_ratings: pd.DataFrame,
    model: NearestNeighbors,
    movie_user_matrix: csr_matrix,
    movie_id_to_index: dict,
    index_to_movie_id: dict,
    neighbors_per_movie: int = 50,
    positive_threshold: float = 4.0,
) -> dict[int, float]:
    """
    Tính điểm ứng viên bằng Item-Based Collaborative Filtering.
    """

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId", "rating"],
    ]

    if user_history.empty:
        return {}

    seen_movie_ids = set(
        user_history["movieId"]
        .astype(int)
        .tolist()
    )

    liked_movies = user_history.loc[
        user_history["rating"] >= positive_threshold
    ]

    if liked_movies.empty:
        liked_movies = (
            user_history
            .sort_values(
                "rating",
                ascending=False,
            )
            .head(5)
        )

    candidate_scores = defaultdict(float)

    number_of_neighbors = min(
        neighbors_per_movie + 1,
        movie_user_matrix.shape[0],
    )

    for row in liked_movies.itertuples(index=False):
        movie_id = int(row.movieId)
        rating = float(row.rating)

        movie_index = movie_id_to_index.get(
            movie_id
        )

        if movie_index is None:
            continue

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

            rating_weight = rating / 5.0

            candidate_scores[neighbor_movie_id] += (
                similarity * rating_weight
            )

    return dict(candidate_scores)


# =========================================================
# TẠO USER PROFILE CONTENT
# =========================================================

def build_content_user_profile(
    user_history: pd.DataFrame,
    movie_feature_matrix: csr_matrix,
    movie_id_to_index: dict,
    positive_threshold: float = 4.0,
) -> csr_matrix | None:
    """
    Tạo vector sở thích nội dung của user.
    """

    liked_movies = user_history.loc[
        user_history["rating"] >= positive_threshold
    ]

    if liked_movies.empty:
        liked_movies = (
            user_history
            .sort_values(
                "rating",
                ascending=False,
            )
            .head(5)
        )

    movie_indices = []
    rating_weights = []

    for row in liked_movies.itertuples(index=False):
        movie_id = int(row.movieId)
        rating = float(row.rating)

        movie_index = movie_id_to_index.get(
            movie_id
        )

        if movie_index is None:
            continue

        movie_indices.append(movie_index)

        # Rating 4 -> 1
        # Rating 4.5 -> 1.5
        # Rating 5 -> 2
        weight = rating - (
            positive_threshold - 1.0
        )

        rating_weights.append(
            max(weight, 0.1)
        )

    if not movie_indices:
        return None

    weights_array = np.asarray(
        rating_weights,
        dtype=np.float32,
    )

    selected_features = movie_feature_matrix[
        np.asarray(
            movie_indices,
            dtype=np.int64,
        )
    ]

    weighted_features = (
        selected_features.multiply(
            weights_array.reshape(-1, 1)
        )
    )

    user_profile = (
        weighted_features.sum(axis=0)
        / weights_array.sum()
    )

    return csr_matrix(
        user_profile,
        dtype=np.float32,
    )


# =========================================================
# TÍNH ĐIỂM CONTENT-BASED
# =========================================================

def get_content_candidate_scores(
    user_id: int,
    train_ratings: pd.DataFrame,
    movie_feature_matrix: csr_matrix,
    movie_id_to_index: dict,
    index_to_movie_id: dict,
    positive_threshold: float = 4.0,
    candidate_limit: int = 500,
) -> dict[int, float]:
    """
    Tính điểm tương đồng giữa user profile
    và toàn bộ phim Content-Based.
    """

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id,
        ["movieId", "rating"],
    ]

    if user_history.empty:
        return {}

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
        return {}

    similarities = cosine_similarity(
        user_profile,
        movie_feature_matrix,
    ).flatten()

    # Loại phim đã xem
    for movie_id in seen_movie_ids:
        movie_index = movie_id_to_index.get(
            movie_id
        )

        if movie_index is not None:
            similarities[movie_index] = -np.inf

    valid_indices = np.where(
        np.isfinite(similarities)
        & (similarities > 0)
    )[0]

    if len(valid_indices) == 0:
        return {}

    number_to_select = min(
        candidate_limit,
        len(valid_indices),
    )

    candidate_positions = np.argpartition(
        similarities,
        -number_to_select,
    )[-number_to_select:]

    ranked_positions = candidate_positions[
        np.argsort(
            similarities[candidate_positions]
        )[::-1]
    ]

    candidate_scores = {}

    for movie_index in ranked_positions:
        movie_index = int(movie_index)

        movie_id = int(
            index_to_movie_id[movie_index]
        )

        candidate_scores[movie_id] = float(
            similarities[movie_index]
        )

    return candidate_scores


# =========================================================
# HYBRID RECOMMENDER
# =========================================================

def recommend_for_user_hybrid(
    user_id: int,
    train_ratings: pd.DataFrame,

    cf_model: NearestNeighbors,
    cf_matrix: csr_matrix,
    cf_movie_id_to_index: dict,
    cf_index_to_movie_id: dict,

    content_matrix: csr_matrix,
    content_movie_id_to_index: dict,
    content_index_to_movie_id: dict,

    top_k: int = 10,
    alpha: float = 0.8,
    neighbors_per_movie: int = 50,
    positive_threshold: float = 4.0,
    content_candidate_limit: int = 500,
) -> list[int]:
    """
    Kết hợp điểm CF và Content-Based.

    hybrid_score =
        alpha * cf_score
        + (1-alpha) * content_score
    """

    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha phải nằm trong khoảng 0 đến 1."
        )

    user_history = train_ratings.loc[
        train_ratings["userId"] == user_id
    ]

    if user_history.empty:
        return []

    seen_movie_ids = set(
        user_history["movieId"]
        .astype(int)
        .tolist()
    )

    cf_scores = get_cf_candidate_scores(
        user_id=user_id,
        train_ratings=train_ratings,
        model=cf_model,
        movie_user_matrix=cf_matrix,
        movie_id_to_index=cf_movie_id_to_index,
        index_to_movie_id=cf_index_to_movie_id,
        neighbors_per_movie=neighbors_per_movie,
        positive_threshold=positive_threshold,
    )

    content_scores = get_content_candidate_scores(
        user_id=user_id,
        train_ratings=train_ratings,
        movie_feature_matrix=content_matrix,
        movie_id_to_index=(
            content_movie_id_to_index
        ),
        index_to_movie_id=(
            content_index_to_movie_id
        ),
        positive_threshold=positive_threshold,
        candidate_limit=content_candidate_limit,
    )

    cf_scores = normalize_scores(cf_scores)

    content_scores = normalize_scores(
        content_scores
    )

    candidate_movie_ids = (
        set(cf_scores.keys())
        | set(content_scores.keys())
    )

    hybrid_scores = {}

    for movie_id in candidate_movie_ids:
        if movie_id in seen_movie_ids:
            continue

        cf_score = cf_scores.get(
            movie_id,
            0.0,
        )

        content_score = content_scores.get(
            movie_id,
            0.0,
        )

        hybrid_score = (
            alpha * cf_score
            + (1.0 - alpha) * content_score
        )

        hybrid_scores[movie_id] = (
            hybrid_score
        )

    ranked_movies = sorted(
        hybrid_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    recommendations = [
        int(movie_id)
        for movie_id, _
        in ranked_movies[:top_k]
    ]

    return recommendations