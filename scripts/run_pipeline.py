from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
 
from data_processing import (
    load_movies_raw,
    load_ratings_raw,
    clean_movies,
    clean_ratings,
    save_processed,
)


DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"


def main() -> None:
    print("=" * 60)
    print("Loading raw datasets...")
    print("=" * 60)

    t0 = time.time()

    # Load raw data
    movies_raw = load_movies_raw()
    ratings_raw = load_ratings_raw()

    n_movies_raw = len(movies_raw)
    n_ratings_raw = len(ratings_raw)

    print(f"Movies : {n_movies_raw:,}")
    print(f"Ratings: {n_ratings_raw:,}")

    # Clean movies
    print("\n" + "=" * 60)
    print("Cleaning movies...")
    print("=" * 60)

    movies_clean = clean_movies(movies_raw)

    # Collaborative Filtering dataset
    print("\n" + "=" * 60)
    print("Cleaning ratings (Collaborative Filtering)...")
    print("=" * 60)

    ratings_cf = clean_ratings(
        ratings_raw,
        min_user_ratings=20,
        min_movie_ratings=50,
    )

    # Content-Based dataset
    print("\n" + "=" * 60)
    print("Cleaning ratings (Content-Based)...")
    print("=" * 60)

    ratings_content = clean_ratings(
        ratings_raw,
        min_user_ratings=20,
        min_movie_ratings=5,
    )

    # Save
    print("\n" + "=" * 60)
    print("Saving processed datasets...")
    print("=" * 60)

    save_processed(
        movies_clean,
        ratings_cf,
        ratings_content,
    )

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)

    print(f"Movies")
    print(f"  Raw   : {n_movies_raw:,}")
    print(f"  Clean : {len(movies_clean):,}")
    print(f"  Retained: {len(movies_clean) / n_movies_raw * 100:.2f}%")

    print()

    print("Ratings (Collaborative Filtering)")
    print(f"  Raw      : {n_ratings_raw:,}")
    print(f"  Filtered : {len(ratings_cf):,}")
    print(f"  Retained : {len(ratings_cf) / n_ratings_raw * 100:.2f}%")

    print()

    print("Ratings (Content-Based)")
    print(f"  Raw      : {n_ratings_raw:,}")
    print(f"  Filtered : {len(ratings_content):,}")
    print(f"  Retained : {len(ratings_content) / n_ratings_raw * 100:.2f}%")

    print()

    print(f"Wall-clock time : {elapsed:.2f} seconds")
    print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()