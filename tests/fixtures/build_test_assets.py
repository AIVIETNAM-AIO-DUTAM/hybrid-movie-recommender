from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"
DATA_DIR = FIXTURE_ROOT / "data" / "processed"
MODEL_DIR = FIXTURE_ROOT / "model"
CF_DIR = MODEL_DIR / "knn_cf"
CONTENT_DIR = MODEL_DIR / "knn_content"


def build_movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "movieId": 101,
                "title": "Toy Story (1995)",
                "genres": "Adventure|Animation|Children|Comedy|Fantasy",
                "year": 1995,
                "genres_list": [
                    "Adventure",
                    "Animation",
                    "Children",
                    "Comedy",
                    "Fantasy",
                ],
                "genres_text": "Adventure Animation Children Comedy Fantasy",
            },
            {
                "movieId": 102,
                "title": "Jumanji (1995)",
                "genres": "Adventure|Children|Fantasy",
                "year": 1995,
                "genres_list": ["Adventure", "Children", "Fantasy"],
                "genres_text": "Adventure Children Fantasy",
            },
            {
                "movieId": 103,
                "title": "Heat (1995)",
                "genres": "Action|Crime|Thriller",
                "year": 1995,
                "genres_list": ["Action", "Crime", "Thriller"],
                "genres_text": "Action Crime Thriller",
            },
            {
                "movieId": 104,
                "title": "Casino (1995)",
                "genres": "Crime|Drama",
                "year": 1995,
                "genres_list": ["Crime", "Drama"],
                "genres_text": "Crime Drama",
            },
            {
                "movieId": 105,
                "title": "Matrix, The (1999)",
                "genres": "Action|Sci-Fi|Thriller",
                "year": 1999,
                "genres_list": ["Action", "Sci-Fi", "Thriller"],
                "genres_text": "Action Sci-Fi Thriller",
            },
            {
                "movieId": 106,
                "title": "Memento (2000)",
                "genres": "Mystery|Thriller",
                "year": 2000,
                "genres_list": ["Mystery", "Thriller"],
                "genres_text": "Mystery Thriller",
            },
        ]
    )


def build_ratings() -> pd.DataFrame:
    rows = [
        # User 1 has liked animation/adventure and action, leaving candidates.
        (1, 101, 5.0, 1_600_000_001),
        (1, 103, 4.5, 1_600_000_101),
        (1, 104, 3.0, 1_600_000_201),
        # Similar users create CF signal.
        (2, 101, 4.5, 1_600_000_011),
        (2, 102, 4.5, 1_600_000_111),
        (2, 103, 3.0, 1_600_000_211),
        (3, 101, 4.0, 1_600_000_021),
        (3, 102, 4.0, 1_600_000_121),
        (3, 106, 3.5, 1_600_000_221),
        (4, 103, 5.0, 1_600_000_031),
        (4, 105, 5.0, 1_600_000_131),
        (4, 106, 4.0, 1_600_000_231),
        (5, 103, 4.5, 1_600_000_041),
        (5, 105, 4.5, 1_600_000_141),
        (5, 104, 3.5, 1_600_000_241),
        (6, 104, 4.0, 1_600_000_051),
        (6, 106, 4.5, 1_600_000_151),
        (6, 105, 4.0, 1_600_000_251),
    ]
    return pd.DataFrame(
        rows,
        columns=["userId", "movieId", "rating", "timestamp"],
    )


def save_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(DATA_DIR / "movies_clean.parquet", index=False)
    ratings.to_parquet(DATA_DIR / "ratings_cf.parquet", index=False)
    ratings.to_parquet(DATA_DIR / "rating_cf_train.parquet", index=False)


def train_cf(ratings: pd.DataFrame) -> None:
    CF_DIR.mkdir(parents=True, exist_ok=True)

    movie_ids = sorted(ratings["movieId"].astype(int).unique())
    user_ids = sorted(ratings["userId"].astype(int).unique())
    movie_id_to_index = {
        movie_id: index for index, movie_id in enumerate(movie_ids)
    }
    user_id_to_index = {
        user_id: index for index, user_id in enumerate(user_ids)
    }

    row_indices = ratings["movieId"].map(movie_id_to_index)
    col_indices = ratings["userId"].map(user_id_to_index)
    matrix = csr_matrix(
        (
            ratings["rating"].astype("float32"),
            (row_indices, col_indices),
        ),
        shape=(len(movie_ids), len(user_ids)),
        dtype="float32",
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(6, len(movie_ids)),
    )
    model.fit(matrix)

    mappings = {
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": {
            index: movie_id for movie_id, index in movie_id_to_index.items()
        },
        "user_id_to_index": user_id_to_index,
        "index_to_user_id": {
            index: user_id for user_id, index in user_id_to_index.items()
        },
    }

    joblib.dump(model, CF_DIR / "knn_cf_model.joblib")
    save_npz(CF_DIR / "movie_user_matrix.npz", matrix)
    joblib.dump(mappings, CF_DIR / "cf_mappings.joblib")
    (CF_DIR / "metadata_cf.json").write_text(
        json.dumps(
            {
                "model_name": "Fixture KNN CF",
                "num_movies": len(movie_ids),
                "num_users": len(user_ids),
                "num_ratings": len(ratings),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train_content(movies: pd.DataFrame) -> None:
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)

    feature_text = (
        movies["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.lower()
        + " "
        + movies["genres_text"].str.lower()
    )
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
        dtype="float32",
    )
    matrix = vectorizer.fit_transform(feature_text)

    movie_ids = movies["movieId"].astype(int).tolist()
    movie_id_to_index = {
        movie_id: index for index, movie_id in enumerate(movie_ids)
    }
    mappings = {
        "movie_id_to_index": movie_id_to_index,
        "index_to_movie_id": {
            index: movie_id for movie_id, index in movie_id_to_index.items()
        },
    }

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=min(6, len(movie_ids)),
    )
    model.fit(matrix)

    joblib.dump(model, CONTENT_DIR / "knn_content_model.joblib")
    joblib.dump(vectorizer, CONTENT_DIR / "tfidf_vectorizer.joblib")
    save_npz(CONTENT_DIR / "movie_feature_matrix.npz", matrix)
    joblib.dump(mappings, CONTENT_DIR / "content_mappings.joblib")
    movies.to_parquet(CONTENT_DIR / "movies_content.parquet", index=False)
    (CONTENT_DIR / "content_metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Fixture KNN Content",
                "num_movies": len(movie_ids),
                "num_features": matrix.shape[1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    movies = build_movies()
    ratings = build_ratings()
    save_data(movies, ratings)
    train_cf(ratings)
    train_content(movies)
    print(f"Fixture data saved to: {DATA_DIR}")
    print(f"Fixture model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
