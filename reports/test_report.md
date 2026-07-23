# Test Report

Owner: QA (Hoàng Đức Kiên) · Status: draft · Last sync: 2026-07-22

## Summary

Mirror của `tests/test_cases.md`. Cập nhật sau khi chạy `pytest tests/ -v`.

| Suite | Tests in code | Cases in `test_cases.md` | Pass | Fail | Skip |
|-------|---------------|--------------------------|------|------|------|
| Simple (S) | 3 (S1, S2, S3) | 3 | 3 | 0 | 0 |
| Content (C) | 2 (C3, C4) | 4 (C1/C2 manual) | 2 | 0 | 0 |
| CF (F) | 3 (F1, F2, F3) | 3 | 3 | 0 | 0 |
| Eval helpers | 1 | — | 1 | 0 | 0 |
| Data pipeline (D) | 3 stubs | 3 | 0 | 0 | 3 (until T10) |
| App smoke (A) | 2 stubs | 2 | 0 | 0 | 2 (until T13) |
| **Total** | **14** | **14 + 2 manual** | **9** | **0** | **5** |

> Con số "Pass" sẽ tăng khi Data (Tâm) và Lead (Tân Dư) hoàn thành T10/T13 → unskip D/A.
> C1/C2 là test thủ công trên data thật, không trong pytest.

## Bugs

| ID | Severity | Description | Owner fix | Status |
|----|----------|-------------|-----------|--------|
| — | — | — | — | — |

| Severity định nghĩa:
| P0 | crash demo / sai data leak (trả phim đã xem) |
| P1 | sai logic core |
| P2 | UX / copy / performance nhẹ |

## Notes

- Cập nhật sau mỗi demo (CN / T4).
- C1/C2 (Toy Story / Heat genre overlap) test thủ công trên app demo.
- E1/E2 (HR@10/NDCG@10 > 0) kiểm tra qua `scripts/run_evaluation.py` ngày T4.
