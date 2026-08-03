# CF Evaluation

Owner: ML B (Loan) + QA (Kiên) · Status: filled · Last sync: 2026-08-03 (D11)

| Metric | Value | Sample size | Notes |
|--------|-------|-------------|-------|
| HR@10 | 0.02 | 200 | leave-last-out by timestamp; evaluated users that CF could score |
| NDCG@10 | 0.0141 | 200 | same sample |
| HR@10_all | 0.02 | 200 | includes cold-start / no-candidate users in denominator |
| NDCG@10_all | 0.0141 | 200 | same |

Source: `evaluation/cf_eval_scores.csv` (produced by `src/evaluation.py::run_evaluation()`, gọi qua `scripts/run_cf_evaluation.py`).

> **⚠️ Protocol khác nhau — đừng so sánh trực tiếp:** `evaluation/model_comparison.csv` + `evaluation_summary.json` (hit_rate@10 = 0.46 cho KNN_CF) dùng **multi-item truth set** (mỗi user ~13 item thật, từ `src/ml/evaluate_model.py` qua `scripts/run_model_evaluation.py`), KHÔNG phải leave-last-out. Bảng trên (HR@10 = 0.01) là leave-last-out single-item. Hai bộ số đo cùng tên metric nhưng khác protocol — final_report chỉ trích dẫn bảng leave-last-out này.


> HR@10 trên sample 200 còn thấp — kỳ vọng với item-based CF leave-last-out trên MovieLens 25M khi chưa tune `min_rating` / neighbor top-K. Đường cải thiện cho sprint sau: tăng `top_k` của item similarity (hiện 100 neighbors/item), giảm `min_rating` từ 4.0 xuống 3.5 để mở rộng liked-pool, hoặc thử user-based CF. Dùng notebook `04_cf_experiments.ipynb` để sweep trước khi khóa số cho final report.

## Reproduce

```bash
source .venv/bin/activate
python -c "from src.data_processing import run_pipeline; run_pipeline()"   # nếu chưa có parquet
python scripts/build_hybrid_artifacts.py    # CF artifacts (top-K sparsified)
python scripts/run_cf_evaluation.py --sample-size 200 --top-k 10   # HR@10 / NDCG@10 cho cf_eval_scores.csv
python scripts/run_hybrid_evaluation.py     # precision/recall/hit_rate cho hybrid (model/knn_cf/)
```

`run_evaluation()` (trong `src/evaluation.py`) tự động:
1. leave-last-out split theo timestamp (per-user) — ổn định với `mergesort` + tie-break movieId
2. build CF trên train (full train split, không sample)
3. sample 200 user eligible, recommend top-10, check HR@10 / NDCG@10

Số liệu ghi vào bảng trên sau khi chạy xong.
