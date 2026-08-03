# CF Evaluation

Owner: ML B (Loan) + QA (Kiên) · Status: filled · Last sync: 2026-07-31

| Metric | Value | Sample size | Notes |
|--------|-------|-------------|-------|
| HR@10 | 0.01 | 200 | leave-last-out by timestamp; evaluated users that CF could score |
| NDCG@10 | 0.0072 | 200 | same sample |
| HR@10_all | 0.01 | 200 | includes cold-start / no-candidate users in denominator |
| NDCG@10_all | 0.0072 | 200 | same |

Source: `evaluation/cf_eval_scores.csv` (produced by `scripts/run_evaluation.py`).

> HR@10 trên sample 200 còn thấp — kỳ vọng với item-based CF leave-last-out trên MovieLens 25M khi chưa tune `min_rating` / neighbor top-K. Dùng notebook `04_cf_experiments.ipynb` để sweep trước khi khóa số cho final report.

## Reproduce

```bash
source .venv/bin/activate
python scripts/run_pipeline.py          # nếu chưa có parquet
python scripts/build_cf_artifacts.py    # CF artifacts (top-K sparsified)
python scripts/run_evaluation.py --sample-size 200 --top-k 10
```

`run_evaluation()` tự động:
1. leave-last-out split theo timestamp (per-user)
2. build CF trên train
3. sample 200 user, recommend top-10, check HR@10 / NDCG@10

Số liệu ghi vào bảng trên sau khi chạy xong.
