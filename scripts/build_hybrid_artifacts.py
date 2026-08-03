"""CLI entrypoint — build CF artifacts (utility matrix + item similarity).

Owner: 18- Thanh Loan (Pipeline) — Tasks T15 + T17.

What this script does
---------------------
1. Load `data/processed/ratings_clean.parquet`
2. Build sparse utility matrix (user × movie) — Task T15
3. Build sparse item-item cosine similarity (movie × movie) — Task T17
4. Save both to `artifacts/`:
   - utility_matrix.npz
   - item_similarity.npz
   - movie_id_maps.pkl  (user_ids, movie_ids, user_to_row, movie_to_col)
5. Print shapes + density so Loan can sanity-check before hand-off to Tech Lead

Usage
-----
    source .venv/bin/activate
    python scripts/build_hybrid_artifacts.py

Acceptance (Tasks T15 + T17)
---------------------------
- `artifacts/utility_matrix.npz` exists, sparse, density < 1%
- `artifacts/item_similarity.npz` exists, sparse
- `artifacts/movie_id_maps.pkl` exists with the 4 keys
- `load_cf_artifacts()` returns a working CFModel
- No OOM on the dev machine
"""

from pathlib import Path
import subprocess

def main():
    root = Path(__file__).resolve().parents[1]
    
    print(" Đang train và build KNN Collaborative Filtering...")
    subprocess.run(["python", str(root / "src" / "ml" / "train_knn_cf.py")], check=True)
    
    print("Đang train và build KNN Content-based...")
    subprocess.run(["python", str(root / "src" / "ml" / "train_knn_content.py")], check=True)
    
    print("Hoàn tất! Model đã được lưu vào model/knn_cf/ và model/knn_content/.")

if __name__ == "__main__":
    main()