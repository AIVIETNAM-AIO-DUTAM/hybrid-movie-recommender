from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EVAL_DIR = ROOT / "evaluation"


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of users to sample for evaluation (default: 200).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="K for HR@K / NDCG@K (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation and save scores."""
    args = parse_args()
    
    # Import các hàm cần thiết từ src
    from data_processing import load_processed
    from evaluation import run_evaluation

    # Load dữ liệu đã xử lý
    print("Đang load dữ liệu đã xử lý...")
    # Load dữ liệu đã xử lý (dùng dấu * để hứng toàn bộ các giá trị trả về)
    print("Đang load dữ liệu đã xử lý...")
    loaded_data = load_processed()
    movies = loaded_data[0]
    ratings = loaded_data[1]

    # Đo thời gian thực thi
    t0 = time.time()

    # Chạy evaluation
    print(f"Đang chạy đánh giá mô hình với sample_size={args.sample_size}, top_k={args.top_k}...")
    scores = run_evaluation(
        ratings, movies,
        sample_size=args.sample_size,
        top_k=args.top_k,
    )

    elapsed = time.time() - t0

    # In kết quả ra stdout
    # In kết quả ra stdout
    print("\n--- KẾT QUẢ ĐÁNH GIÁ COLLABORATIVE FILTERING ---")
    print(scores.to_string(index=False))
    
    # In thời gian hoàn thành
    print(f"\nHoàn thành đánh giá trong {elapsed:.2f} giây.")

    REPORTS_DIR = ROOT / "reports"
    # REPORTS_DIR = ROOT / "evaluation"

    # Đảm bảo thư mục tồn tại
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Lưu file CSV vào thư mục evaluation
    csv_path = EVAL_DIR / "cf_eval_scores.csv"
    scores.to_csv(csv_path, index=False)
    print(f"Đã lưu kết quả CSV tại: {csv_path}")
if __name__ == "__main__":
    main()