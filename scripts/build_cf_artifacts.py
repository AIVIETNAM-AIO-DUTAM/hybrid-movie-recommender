from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Build and persist CF artifacts."""
    from data_processing import load_processed
    from recommender_cf import (
        build_utility_matrix,
        build_item_similarity,
        CFModel,
        save_cf_artifacts,
    )

    print("Đang load dữ liệu đã xử lý...")
    loaded_data = load_processed()
    movies = loaded_data[0]
    ratings = loaded_data[1]

    t0 = time.time()

    print("Đang xây dựng utility matrix (Task T15)...")
    utility, user_ids, movie_ids, u2r, m2c = build_utility_matrix(ratings)
    
    # Tính toán thông tin shape, số phần tử khác không và độ mật độ (density)
    shape = utility.shape
    nnz = utility.nnz
    total_elements = shape[0] * shape[1]
    density = (nnz / total_elements) * 100 if total_elements > 0 else 0
    print(f"-> Utility matrix shape: {shape}, nnz: {nnz}, density: {density:.4f}%")

    print("Đang xây dựng ma trận độ tương đồng item-item (Task T17)...")
    item_sim = build_item_similarity(utility)
    
    sim_shape = item_sim.shape
    sim_nnz = item_sim.nnz
    sim_total = sim_shape[0] * sim_shape[1]
    sim_density = (sim_nnz / sim_total) * 100 if sim_total > 0 else 0
    print(f"-> Item similarity matrix shape: {sim_shape}, nnz: {sim_nnz}, density: {sim_density:.4f}%")

    print("Đang khởi tạo CFModel và lưu các artifacts...")
    model = CFModel(utility, item_sim, user_ids, movie_ids, u2r, m2c)
    save_cf_artifacts(model)

    elapsed = time.time() - t0
    print(f"\nHoàn thành xây dựng và lưu CF artifacts trong {elapsed:.2f} giây.")


if __name__ == "__main__":
    main()