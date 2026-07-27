from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# File hiện tại nằm tại:
# hybrid-movie-recommender/src/ml/train_knn_cf.py

# parents[0] = src/ml
# parents[1] = src
# parents[2] = hybrid-movie-recommender
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR = BASE_DIR / "model" / "knn_cf"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RATINGS_PATH = DATA_DIR / "ratings_cf.parquet"

TRAIN_PATH = DATA_DIR / "rating_cf_train.parquet"

TEST_PATH = DATA_DIR / "rating_cf_test.parquet"


# =========================================================
# 2. ĐỌC DỮ LIỆU
# =========================================================

def load_ratings(
    input_path: Path,
) -> pd.DataFrame:
    """
    Đọc dữ liệu rating dùng cho Collaborative Filtering.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu:\n{input_path}"
        )

    ratings = pd.read_parquet(input_path)

    required_columns = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    missing_columns = (
        required_columns - set(ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            f"File dữ liệu thiếu các cột: "
            f"{missing_columns}"
        )

    ratings = ratings[
        [
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ]
    ].copy()

    ratings = ratings.dropna(
        subset=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ]
    )

    ratings["userId"] = (
        ratings["userId"].astype(np.int64)
    )

    ratings["movieId"] = (
        ratings["movieId"].astype(np.int64)
    )

    ratings["rating"] = (
        ratings["rating"].astype(np.float32)
    )

    ratings["timestamp"] = (
        ratings["timestamp"].astype(np.int64)
    )

    # Loại bỏ trường hợp một user đánh giá một phim nhiều lần
    # Giữ lại rating mới nhất
    ratings = (
        ratings
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["userId", "movieId"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    print(
        f"Số ratings sau khi đọc: "
        f"{len(ratings):,}"
    )

    print(
        f"Số users: "
        f"{ratings['userId'].nunique():,}"
    )

    print(
        f"Số movies: "
        f"{ratings['movieId'].nunique():,}"
    )

    return ratings


# =========================================================
# 3. CHIA TRAIN/TEST THEO THỜI GIAN
# =========================================================

def train_test_split_by_time(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_test: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia train/test theo thời gian của từng user.

    Với mỗi user:
    - Rating cũ hơn được đưa vào train.
    - Rating mới nhất được đưa vào test.
    """

    required_columns = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    missing_columns = (
        required_columns - set(ratings.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột: {missing_columns}"
        )

    if not 0 < test_ratio < 1:
        raise ValueError(
            "test_ratio phải nằm trong khoảng (0, 1)"
        )

    if min_test < 1:
        raise ValueError(
            "min_test phải lớn hơn hoặc bằng 1"
        )

    data = ratings.copy()

    # Sắp xếp rating của từng user theo thời gian
    data = data.sort_values(
        by=["userId", "timestamp"]
    ).reset_index(drop=True)

    # Vị trí rating trong lịch sử của từng user
    data["position"] = (
        data.groupby("userId").cumcount()
    )

    # Tổng số rating của từng user
    data["user_rating_count"] = (
        data.groupby("userId")["movieId"]
        .transform("size")
    )

    # Số rating được đưa vào tập test
    data["n_test"] = np.maximum(
        np.floor(
            data["user_rating_count"]
            * test_ratio
        ).astype(int),
        min_test,
    )

    # Mỗi user phải còn ít nhất một rating trong train
    data["n_test"] = np.minimum(
        data["n_test"],
        data["user_rating_count"] - 1,
    )

    # Những user chỉ có một rating sẽ có n_test = 0
    data["n_test"] = np.maximum(
        data["n_test"],
        0,
    )

    test_mask = (
        data["position"]
        >= (
            data["user_rating_count"]
            - data["n_test"]
        )
    ) & (data["n_test"] > 0)

    train = data.loc[~test_mask].copy()
    test = data.loc[test_mask].copy()

    helper_columns = [
        "position",
        "user_rating_count",
        "n_test",
    ]

    train.drop(
        columns=helper_columns,
        inplace=True,
    )

    test.drop(
        columns=helper_columns,
        inplace=True,
    )

    train.reset_index(
        drop=True,
        inplace=True,
    )

    test.reset_index(
        drop=True,
        inplace=True,
    )

    return train, test


# =========================================================
# 4. TẠO MA TRẬN MOVIE × USER
# =========================================================

def build_movie_user_matrix(
    ratings: pd.DataFrame,
) -> tuple[
    csr_matrix,
    np.ndarray,
    np.ndarray,
]:
    """
    Tạo ma trận sparse:

        dòng: movieId
        cột: userId
        giá trị: rating
    """

    if ratings.empty:
        raise ValueError(
            "Dữ liệu train không có rating."
        )

    movie_codes, movie_ids = pd.factorize(
        ratings["movieId"],
        sort=True,
    )

    user_codes, user_ids = pd.factorize(
        ratings["userId"],
        sort=True,
    )

    movie_user_matrix = csr_matrix(
        (
            ratings["rating"].to_numpy(
                dtype=np.float32
            ),
            (
                movie_codes,
                user_codes,
            ),
        ),
        shape=(
            len(movie_ids),
            len(user_ids),
        ),
        dtype=np.float32,
    )

    movie_ids = movie_ids.to_numpy(
        dtype=np.int64
    )

    user_ids = user_ids.to_numpy(
        dtype=np.int64
    )

    return (
        movie_user_matrix,
        movie_ids,
        user_ids,
    )


# =========================================================
# 5. TẠO MAPPING
# =========================================================

def create_mappings(
    movie_ids: np.ndarray,
    user_ids: np.ndarray,
) -> dict:
    """
    Tạo mapping giữa ID thật và vị trí trong ma trận.
    """

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

    user_id_to_index = {
        int(user_id): int(index)
        for index, user_id
        in enumerate(user_ids)
    }

    index_to_user_id = {
        int(index): int(user_id)
        for index, user_id
        in enumerate(user_ids)
    }

    return {
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": index_to_movie_id,
        "user_id_to_index": user_id_to_index,
        "index_to_user_id": index_to_user_id,
    }


# =========================================================
# 6. TRAIN KNN COLLABORATIVE FILTERING
# =========================================================

def train_knn_cf(
    movie_user_matrix: csr_matrix,
    n_neighbors: int = 20,
) -> NearestNeighbors:
    """
    Train Item-Based KNN Collaborative Filtering.
    """

    number_of_movies = (
        movie_user_matrix.shape[0]
    )

    if number_of_movies == 0:
        raise ValueError(
            "Ma trận movie-user không có phim."
        )

    n_neighbors = min(
        n_neighbors,
        number_of_movies,
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors,
        n_jobs=-1,
    )

    model.fit(movie_user_matrix)

    print(
        f"Đã train KNN-CF với "
        f"n_neighbors={n_neighbors}"
    )

    return model


# =========================================================
# 7. LƯU ARTIFACTS
# =========================================================

def save_artifacts(
    model: NearestNeighbors,
    movie_user_matrix: csr_matrix,
    mappings: dict,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    n_neighbors: int,
    test_ratio: float,
) -> None:
    """
    Lưu model, sparse matrix, mappings và metadata.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Lưu KNN model
    joblib.dump(
        model,
        MODEL_DIR / "knn_cf_model.joblib",
    )

    # Lưu ma trận movie-user
    save_npz(
        MODEL_DIR / "movie_user_matrix.npz",
        movie_user_matrix,
    )

    # Lưu mapping
    joblib.dump(
        mappings,
        MODEL_DIR / "cf_mappings.joblib",
    )

    metadata = {
        "model_name": (
            "Item-Based KNN Collaborative Filtering"
        ),
        "model_type": "NearestNeighbors",
        "metric": "cosine",
        "algorithm": "brute",
        "n_neighbors": int(n_neighbors),
        "num_movies": int(
            movie_user_matrix.shape[0]
        ),
        "num_users": int(
            movie_user_matrix.shape[1]
        ),
        "num_train_ratings": int(
            len(train_ratings)
        ),
        "num_test_ratings": int(
            len(test_ratings)
        ),
        "matrix_shape": [
            int(movie_user_matrix.shape[0]),
            int(movie_user_matrix.shape[1]),
        ],
        "matrix_nonzero_values": int(
            movie_user_matrix.nnz
        ),
        "test_ratio": float(test_ratio),
        "positive_rating_threshold": 4.0,
    }

    with open(
        MODEL_DIR / "metadata_cf.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\nĐã lưu các file:")

    for file_path in sorted(
        MODEL_DIR.iterdir()
    ):
        print(f"- {file_path.name}")


# =========================================================
# 8. KIỂM TRA GỢI Ý PHIM TƯƠNG TỰ
# =========================================================

def recommend_similar_movies(
    movie_id: int,
    model: NearestNeighbors,
    movie_user_matrix: csr_matrix,
    mappings: dict,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Tìm các phim tương tự một movieId bằng KNN-CF.
    """

    movie_id_to_index = (
        mappings["movie_id_to_index"]
    )

    index_to_movie_id = (
        mappings["index_to_movie_id"]
    )

    if movie_id not in movie_id_to_index:
        raise ValueError(
            f"Không tìm thấy movieId={movie_id} "
            "trong dữ liệu train."
        )

    movie_index = movie_id_to_index[movie_id]

    number_of_neighbors = min(
        top_k + 1,
        movie_user_matrix.shape[0],
    )

    distances, indices = model.kneighbors(
        movie_user_matrix[movie_index],
        n_neighbors=number_of_neighbors,
    )

    recommendations = []

    for neighbor_index, distance in zip(
        indices.flatten(),
        distances.flatten(),
    ):
        neighbor_movie_id = (
            index_to_movie_id[
                int(neighbor_index)
            ]
        )

        # Bỏ chính phim đầu vào
        if neighbor_movie_id == movie_id:
            continue

        recommendations.append(
            {
                "movieId": int(
                    neighbor_movie_id
                ),
                "cf_score": (
                    1.0 - float(distance)
                ),
            }
        )

        if len(recommendations) >= top_k:
            break

    return pd.DataFrame(recommendations)


# =========================================================
# 9. HÀM TRAIN CHÍNH
# =========================================================

def train_model() -> None:
    """
    Quy trình train KNN Collaborative Filtering.
    """

    test_ratio = 0.2
    n_neighbors = 20

    print("=" * 60)
    print("TRAIN ITEM-BASED KNN COLLABORATIVE FILTERING")
    print("=" * 60)

    print(f"\nProject root:\n{BASE_DIR}")
    print(f"\nInput path:\n{RATINGS_PATH}")
    print(f"\nModel directory:\n{MODEL_DIR}")

    # Bước 1: Đọc dữ liệu
    print("\n[1/6] Loading ratings...")

    ratings_cf = load_ratings(
        RATINGS_PATH
    )

    # Bước 2: Chia train/test
    print("\n[2/6] Splitting train/test...")

    train_rating_cf, test_rating_cf = (
        train_test_split_by_time(
            ratings=ratings_cf,
            test_ratio=test_ratio,
            min_test=1,
        )
    )

    print(
        f"Train rows: "
        f"{len(train_rating_cf):,}"
    )

    print(
        f"Test rows: "
        f"{len(test_rating_cf):,}"
    )

    # Lưu train/test để đánh giá về sau
    train_rating_cf.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    test_rating_cf.to_parquet(
        TEST_PATH,
        index=False,
    )

    print(f"Đã lưu train tại:\n{TRAIN_PATH}")
    print(f"Đã lưu test tại:\n{TEST_PATH}")

    # Bước 3: Tạo sparse matrix
    print(
        "\n[3/6] Creating sparse "
        "movie-user matrix..."
    )

    (
        movie_user_matrix,
        movie_ids,
        user_ids,
    ) = build_movie_user_matrix(
        train_rating_cf
    )

    print(
        f"Matrix shape: "
        f"{movie_user_matrix.shape}"
    )

    print(
        f"Non-zero ratings: "
        f"{movie_user_matrix.nnz:,}"
    )

    # Bước 4: Tạo mappings
    print("\n[4/6] Creating mappings...")

    mappings = create_mappings(
        movie_ids=movie_ids,
        user_ids=user_ids,
    )

    print(
        f"Movie mappings: "
        f"{len(mappings['movie_id_to_index']):,}"
    )

    print(
        f"User mappings: "
        f"{len(mappings['user_id_to_index']):,}"
    )

    # Bước 5: Train KNN
    print("\n[5/6] Training KNN-CF...")

    knn_cf = train_knn_cf(
        movie_user_matrix=movie_user_matrix,
        n_neighbors=n_neighbors,
    )

    # Bước 6: Lưu model
    print("\n[6/6] Saving artifacts...")

    save_artifacts(
        model=knn_cf,
        movie_user_matrix=movie_user_matrix,
        mappings=mappings,
        train_ratings=train_rating_cf,
        test_ratings=test_rating_cf,
        n_neighbors=n_neighbors,
        test_ratio=test_ratio,
    )

    print("\n" + "=" * 60)
    print("TRAIN KNN-CF HOÀN TẤT")
    print("=" * 60)

    print(
        f"\nArtifacts được lưu tại:\n"
        f"{MODEL_DIR}"
    )


if __name__ == "__main__":
    train_model()