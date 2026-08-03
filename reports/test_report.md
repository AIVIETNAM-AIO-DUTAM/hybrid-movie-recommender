# Test Report

Owner: QA (Hoàng Đức Kiên) · Status: updated · Last sync: 2026-07-31

## Summary

Mirror của `tests/test_cases.md`. Cập nhật sau khi chạy `pytest tests/ -v`.

| Suite | Tests in code | Cases in `test_cases.md` | Pass | Fail | Skip |
|-------|---------------|--------------------------|------|------|------|
| Simple (S) | 3 (S1, S2, S3) | 3 | 3 | 0 | 0 |
| Content (C) | 4 (C3, C4 + overlap) | 4 (C1/C2 manual) | 4 | 0 | 0 |
| CF (F) | 4 (F1–F4) | 4 | 4 | 0 | 0 |
| Eval helpers | 1 | — | 1 | 0 | 0 |
| Data pipeline (D) | 3 (D1–D3) | 3 | 3 | 0 | 0 |
| App smoke (A) | 2 (A1, A2) | 2 | 2* | 0 | 0* |
| **Total** | **17** | **14 + 2 manual** | **17** | **0** | **0** |

\* A2 cần `data/processed/movies_clean.parquet` (đã có sau `run_pipeline.py`). A1 luôn chạy được (monkeypatch missing parquet).

> C1/C2 là test thủ công trên data thật (Toy Story / Heat genre overlap), không trong pytest — dùng tab Content trong `streamlit run src/app.py`.

## Bugs

| ID | Severity | Description | Owner fix | Status |
|----|----------|-------------|-----------|--------|
| B1 | P1 | `load_processed()` trả 3-tuple nhưng `src/app.py` unpack 2 giá trị → crash demo | Tân Dư | Fixed (D8) |
| B2 | P1 | Tests D1–D3 trỏ `ratings_clean.parquet` trong khi pipeline ghi `ratings_cf.parquet` | Kiên / Tâm | Fixed (alias + test update) |
| B3 | P2 | `genre_overlap_at_k` thiếu so với plan T16 / DoD Content | Duong | Fixed (D8) |
| B4 | P1 | Full item-similarity trên 25M dễ OOM | Loan | Fixed (top-K sparsify + chunk) |

| Severity định nghĩa:
| P0 | crash demo / sai data leak (trả phim đã xem) |
| P1 | sai logic core |
| P2 | UX / copy / performance nhẹ |

## Notes

- Unskip D1–D3 / A1–A2 hoàn tất trên branch `fix/sprint-gaps-d8`.
- E1/E2 (HR@10/NDCG@10 > 0) đã có số liệu trong `evaluation/cf_eval_scores.csv` + `reports/cf_evaluation.md`.
- C1/C2 vẫn manual trên app demo trước APP FREEZE 01/08.
