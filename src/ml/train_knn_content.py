from pathlib import Path
import json

import joblib
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================

# Thư mục chứa file train_knn_content.py
CURRENT_DIR = Path(__file__).resolve().parent

# Thư mục gốc hybrid-movie-recommender
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# File dữ liệu đầu vào
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "movies_clean.parquet"
)

# Thư mục lưu mô hình Content-Based
OUTPUT_DIR = (
    PROJECT_ROOT
    / "model"
    / "knn_content"
)


# =========================================================
# 2. ĐỌC DỮ LIỆU
# =========================================================

def load_movies(input_path: Path) -> pd.DataFrame:
    """
    Đọc file movies_clean.parquet và kiểm tra các cột cần thiết.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu:\n{input_path}"
        )

    movies = pd.read_parquet(input_path)

    required_columns = {
        "movieId",
        "title",
        "genres"
    }

    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"File dữ liệu thiếu các cột: {missing_columns}"
        )

    print(f"Số lượng phim ban đầu: {len(movies):,}")

    return movies


# =========================================================
# 3. TIỀN XỬ LÝ DỮ LIỆU
# =========================================================

def prepare_features(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch title và genres để tạo feature_text.

    feature_text gồm:
    - clean_title
    - genres_text lặp lại 2 lần để tăng trọng số thể loại
    """

    movies = movies.copy()

    # Chỉ giữ các cột cần thiết
    movies = movies[
        [
            "movieId",
            "title",
            "genres"
        ]
    ]

    # Loại bỏ dòng thiếu movieId
    movies = movies.dropna(subset=["movieId"])

    # Chuyển movieId về số nguyên
    movies["movieId"] = movies["movieId"].astype(int)

    # Loại bỏ movieId bị trùng
    movies = movies.drop_duplicates(
        subset=["movieId"],
        keep="first"
    )

    # Reset index khớp với ma trận TF-IDF
    movies = movies.reset_index(drop=True)

    # Xử lý title
    movies["title"] = (
        movies["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Xóa năm ở cuối tiêu đề
    # Ví dụ: The Matrix (1999) -> The Matrix
    movies["clean_title"] = (
        movies["title"]
        .str.replace(
            r"\s*\(\d{4}\)\s*$",
            "",
            regex=True
        )
        .str.lower()
        .str.strip()
    )

    # Chuyển genres thành văn bản
    # Ví dụ:
    # Action|Sci-Fi|Thriller
    # ->
    # action sci-fi thriller
    movies["genres_text"] = (
        movies["genres"]
        .fillna("")
        .astype(str)
        .str.replace(
            "(no genres listed)",
            "",
            regex=False
        )
        .str.replace(
            "|",
            " ",
            regex=False
        )
        .str.lower()
        .str.strip()
    )

    # Giữ title và lặp genres để tăng trọng số cho thể loại
    movies["feature_text"] = (
        movies["clean_title"]
        + " "
        + movies["genres_text"]
        + " "
        + movies["genres_text"]
    )

    # Xóa khoảng trắng dư
    movies["feature_text"] = (
        movies["feature_text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    # Loại bỏ phim không có bất kỳ thông tin nào
    movies = movies[
        movies["feature_text"].str.len() > 0
    ].reset_index(drop=True)

    print(f"Số lượng phim sau xử lý: {len(movies):,}")

    return movies


# =========================================================
# 4. TẠO MA TRẬN TF-IDF
# =========================================================

def create_tfidf_matrix(
    movies: pd.DataFrame
) -> tuple[TfidfVectorizer, object]:
    """
    Chuyển feature_text thành ma trận TF-IDF.
    """

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",

        # Dùng unigram và bigram
        ngram_range=(1, 2),

        # Từ phải xuất hiện ít nhất trong 2 phim
        min_df=2,

        # Giới hạn số feature để tránh ma trận quá lớn
        max_features=30000,

        # Chuẩn hóa vector để dùng cosine
        norm="l2",

        dtype="float32"
    )

    movie_feature_matrix = (
        tfidf_vectorizer.fit_transform(
            movies["feature_text"]
        )
    )

    print(
        "Kích thước ma trận TF-IDF:",
        movie_feature_matrix.shape
    )

    print(
        f"Số phần tử khác 0: "
        f"{movie_feature_matrix.nnz:,}"
    )

    return tfidf_vectorizer, movie_feature_matrix


# =========================================================
# 5. TẠO MAPPING
# =========================================================

def create_mappings(
    movies: pd.DataFrame
) -> tuple[dict, dict]:
    """
    Tạo mapping:

    movie_id_to_index:
        movieId thật -> vị trí dòng trong ma trận TF-IDF

    index_to_movie_id:
        vị trí dòng trong ma trận -> movieId thật
    """

    movie_id_to_index = {
        int(movie_id): int(index)
        for index, movie_id
        in enumerate(movies["movieId"])
    }

    index_to_movie_id = {
        int(index): int(movie_id)
        for index, movie_id
        in enumerate(movies["movieId"])
    }

    return movie_id_to_index, index_to_movie_id


# =========================================================
# 6. TRAIN KNN CONTENT-BASED
# =========================================================

def train_knn_content(
    movie_feature_matrix,
    n_neighbors: int = 50
) -> NearestNeighbors:
    """
    Train KNN Content-Based bằng cosine distance.
    """

    number_of_movies = movie_feature_matrix.shape[0]

    if number_of_movies == 0:
        raise ValueError(
            "Ma trận TF-IDF không có phim nào."
        )

    # Không để n_neighbors lớn hơn số lượng phim
    n_neighbors = min(
        n_neighbors,
        number_of_movies
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=n_neighbors,
        n_jobs=-1
    )

    model.fit(movie_feature_matrix)

    print(
        f"Đã train KNN với n_neighbors={n_neighbors}"
    )

    return model


# =========================================================
# 7. LƯU MODEL VÀ CÁC FILE LIÊN QUAN
# =========================================================

def save_artifacts(
    output_dir: Path,
    movies: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    movie_feature_matrix,
    model: NearestNeighbors,
    movie_id_to_index: dict,
    index_to_movie_id: dict
) -> None:
    """
    Lưu model, TF-IDF, ma trận, mapping và metadata.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Lưu KNN model
    joblib.dump(
        model,
        output_dir / "knn_content_model.joblib"
    )

    # Lưu TF-IDF vectorizer
    joblib.dump(
        tfidf_vectorizer,
        output_dir / "tfidf_vectorizer.joblib"
    )

    # Lưu ma trận sparse TF-IDF
    save_npz(
        output_dir / "movie_feature_matrix.npz",
        movie_feature_matrix
    )

    # Lưu mapping
    joblib.dump(
        {
            "movie_id_to_index": movie_id_to_index,
            "index_to_movie_id": index_to_movie_id
        },
        output_dir / "content_mappings.joblib"
    )

    # Lưu thông tin phim đã xử lý
    movies[
        [
            "movieId",
            "title",
            "genres",
            "clean_title",
            "genres_text",
            "feature_text"
        ]
    ].to_parquet(
        output_dir / "movies_content.parquet",
        index=False
    )

    # Lưu metadata
    metadata = {
        "model_name": "KNN Content-Based",
        "model_type": "NearestNeighbors",
        "metric": "cosine",
        "algorithm": "brute",
        "number_of_movies": int(
            movie_feature_matrix.shape[0]
        ),
        "number_of_features": int(
            movie_feature_matrix.shape[1]
        ),
        "number_of_nonzero_values": int(
            movie_feature_matrix.nnz
        ),
        "feature_source": [
            "clean_title",
            "genres_text",
            "genres_text"
        ],
        "tfidf_parameters": {
            "stop_words": "english",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 30000,
            "norm": "l2"
        }
    }

    with open(
        output_dir / "content_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nĐã lưu các file:")

    for file_path in sorted(output_dir.iterdir()):
        print(f"- {file_path.name}")


# =========================================================
# 8. HÀM KIỂM TRA GỢI Ý
# =========================================================

def test_recommendation(
    movie_id: int,
    movies: pd.DataFrame,
    movie_feature_matrix,
    model: NearestNeighbors,
    movie_id_to_index: dict,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Kiểm tra nhanh mô hình sau khi train.
    """

    if movie_id not in movie_id_to_index:
        raise ValueError(
            f"Không tìm thấy movieId={movie_id}"
        )

    movie_index = movie_id_to_index[movie_id]

    number_of_neighbors = min(
        top_k + 1,
        len(movies)
    )

    distances, indices = model.kneighbors(
        movie_feature_matrix[movie_index],
        n_neighbors=number_of_neighbors
    )

    recommendations = []

    for neighbor_index, distance in zip(
        indices.flatten(),
        distances.flatten()
    ):
        neighbor_movie_id = int(
            movies.iloc[neighbor_index]["movieId"]
        )

        # Bỏ chính bộ phim đầu vào
        if neighbor_movie_id == movie_id:
            continue

        content_score = 1 - float(distance)

        recommendations.append(
            {
                "movieId": neighbor_movie_id,
                "title": movies.iloc[
                    neighbor_index
                ]["title"],
                "genres": movies.iloc[
                    neighbor_index
                ]["genres"],
                "content_score": content_score
            }
        )

        if len(recommendations) >= top_k:
            break

    return pd.DataFrame(recommendations)


# =========================================================
# 9. MAIN
# =========================================================

def main() -> None:
    print("=" * 60)
    print("TRAIN KNN CONTENT-BASED MOVIE RECOMMENDER")
    print("=" * 60)

    print(f"\nProject root:\n{PROJECT_ROOT}")
    print(f"\nInput path:\n{INPUT_PATH}")
    print(f"\nOutput directory:\n{OUTPUT_DIR}")

    # Bước 1: Đọc dữ liệu
    print("\n[1/6] Đang đọc dữ liệu...")

    movies = load_movies(INPUT_PATH)

    # Bước 2: Tiền xử lý
    print("\n[2/6] Đang tạo feature_text...")

    movies = prepare_features(movies)

    # Bước 3: Tạo TF-IDF
    print("\n[3/6] Đang tạo ma trận TF-IDF...")

    tfidf_vectorizer, movie_feature_matrix = (
        create_tfidf_matrix(movies)
    )

    # Bước 4: Tạo mapping
    print("\n[4/6] Đang tạo mapping...")

    movie_id_to_index, index_to_movie_id = (
        create_mappings(movies)
    )

    print(
        f"Số movieId trong mapping: "
        f"{len(movie_id_to_index):,}"
    )

    # Bước 5: Train KNN
    print("\n[5/6] Đang train KNN...")

    model = train_knn_content(
        movie_feature_matrix=movie_feature_matrix,
        n_neighbors=50
    )

    # Bước 6: Lưu
    print("\n[6/6] Đang lưu model...")

    save_artifacts(
        output_dir=OUTPUT_DIR,
        movies=movies,
        tfidf_vectorizer=tfidf_vectorizer,
        movie_feature_matrix=movie_feature_matrix,
        model=model,
        movie_id_to_index=movie_id_to_index,
        index_to_movie_id=index_to_movie_id
    )

    # Kiểm tra với The Matrix nếu có movieId 2571
    test_movie_id = 2571

    if test_movie_id in movie_id_to_index:
        print("\n" + "=" * 60)
        print(
            f"KIỂM TRA GỢI Ý CHO movieId={test_movie_id}"
        )
        print("=" * 60)

        recommendations = test_recommendation(
            movie_id=test_movie_id,
            movies=movies,
            movie_feature_matrix=movie_feature_matrix,
            model=model,
            movie_id_to_index=movie_id_to_index,
            top_k=10
        )

        print(
            recommendations.to_string(
                index=False
            )
        )

    print("\n" + "=" * 60)
    print("TRAIN CONTENT-BASED HOÀN TẤT")
    print("=" * 60)

    print(f"\nModel được lưu tại:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()