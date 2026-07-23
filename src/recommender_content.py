"""Content-based Recommender — genre cosine. Owned by ML A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentModel:
    movies: pd.DataFrame
    vectorizer: CountVectorizer
    genre_matrix: object  # sparse (n_movies x n_genres)


NO_GENRES_SENTINEL = "(no genres listed)"


def _genres_text_from_genres(genres: pd.Series) -> pd.Series:
    """Build genres_text from raw `genres` column, filtering the sentinel.

    Mirrors data_processing.clean_movies so callers without a pre-built
    `genres_text` column still get the sentinel-free vocabulary that the
    production pipeline produces. Without this, CountVectorizer would learn
    spurious tokens "(no", "genres", "listed)".
    """
    def _split_join(g):
        if not isinstance(g, str):
            return ""
        parts = [p for p in g.split("|") if p and p != NO_GENRES_SENTINEL]
        return " ".join(parts)
    return genres.apply(_split_join)


def build_content_model(movies: pd.DataFrame) -> ContentModel:
    df = movies.copy()
    if "genres_text" not in df.columns:
        df["genres_text"] = _genres_text_from_genres(df["genres"])
    # token_pattern=r"\S+" keeps hyphenated genres like "Sci-Fi" as a single
    # token. Default tokenizer "\b\w\w+\b" would split "Sci-Fi" -> ["sci","fi"]
    # and corrupt the vocabulary used by cosine similarity.
    vectorizer = CountVectorizer(token_pattern=r"\S+")
    genre_matrix = vectorizer.fit_transform(df["genres_text"].fillna(""))
    # Keep genre_matrix sparse. With MovieLens 25M (~62k movies) a dense
    # (62k x 62k) similarity matrix would need ~30GB RAM, so we compute
    # similarity on-demand per query instead of pre-materializing it.
    return ContentModel(
        movies=df.reset_index(drop=True),
        vectorizer=vectorizer,
        genre_matrix=genre_matrix,
    )


def _resolve_index(model: ContentModel, movie: Union[int, str]) -> int:
    movies = model.movies
    if isinstance(movie, (int, np.integer)):
        hits = movies.index[movies["movieId"] == int(movie)].tolist()
        if not hits:
            raise ValueError(f"movieId not found: {movie}")
        return hits[0]
    # Exact match first — deterministic and unambiguous.
    exact = movies.index[movies["title"] == movie].tolist()
    if exact:
        return exact[0]
    # Partial match: among candidates with parenthesized year, prefer the
    # SHORTEST title (closest to the literal query) so "Matrix" maps to
    # "Matrix (1999)" rather than "Matrix Reloaded (2003)" or the longer
    # "The Matrix (1999)". Falls back to any partial if no year-form match.
    partial = movies.index[
        movies["title"].str.contains(str(movie), case=False, regex=False, na=False)
    ].tolist()
    if not partial:
        raise ValueError(f"title not found: {movie}")
    candidates_with_year = [
        i for i in partial
        if "(" in movies.at[i, "title"] and ")" in movies.at[i, "title"]
    ]
    pool = candidates_with_year or partial
    # Sort by (title length asc, title asc) for deterministic preference:
    # shorter canonical title wins ties over longer disambiguated ones.
    return sorted(pool, key=lambda i: (len(movies.at[i, "title"]), movies.at[i, "title"]))[0]


def recommend_similar_movies(
    model: ContentModel,
    movie: Union[int, str],
    top_k: int = 10,
) -> pd.DataFrame:
    idx = _resolve_index(model, movie)
    # Guard: query movie with empty genres -> cosine is undefined (zero norm).
    # Returning arbitrary top-K would surface noise as "similar" movies.
    if model.genre_matrix[idx].nnz == 0:
        raise ValueError(
            f"movie at index {idx} has no genres; cannot compute similarity"
        )
    # Cosine similarity between the query movie and all others in one shot.
    # genre_matrix is sparse -> cosine_similarity returns a dense (1, n) array.
    scores = cosine_similarity(
        model.genre_matrix[idx], model.genre_matrix
    ).ravel()
    scores = scores.astype(float)
    scores[idx] = -1.0  # exclude self
    # argsort(-scores) already returns indices in descending-score order
    # (best first). Do NOT append [::-1] — that flips to ascending (worst).
    top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    rows = model.movies.iloc[top_idx][["movieId", "title", "genres"]].copy()
    rows["similarity"] = scores[top_idx]
    rows["shared_genres"] = rows["genres"].apply(
        lambda g: _shared_genres(model.movies.iloc[idx]["genres"], g)
    )
    return rows.reset_index(drop=True)


def _shared_genres(a: Optional[str], b: Optional[str]) -> str:
    sa = set((a or "").split("|"))
    sb = set((b or "").split("|"))
    shared = sorted(sa & sb - {""})
    return "|".join(shared)
