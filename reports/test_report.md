# Test Report

Owner: QA (Hoàng Đức Kiên) · Status: updated · Last sync: 2026-08-03

## Summary

Mirror của `tests/test_cases.md`. Cập nhật sau khi chạy:

```bash
.venv313/bin/python -m pytest tests -v
```

| Suite | Tests in code | Cases in `test_cases.md` | Pass | Fail | Skip |
|-------|---------------|--------------------------|------|------|------|
| Simple (S) | 3 (S1, S2, S3) | 3 | 3 | 0 | 0 |
| Content (C) | 2 (C3, C4) | 4 (C1/C2 manual) | 2 | 0 | 0 |
| CF (F) | 4 (F1, F2, F3, F4) | 4 | 4 | 0 | 0 |
| Eval helpers | 1 | — | 1 | 0 | 0 |
| Data pipeline (D) | 3 | 3 | 3 | 0 | 0 |
| App smoke (A) | 2 | 2 | 2 | 0 | 0 |
| **Total** | **15** | **15 + 2 manual** | **15** | **0** | **0** |

> D/A hiện chạy bằng fixture nhỏ trong `tests/fixtures/`, không cần full dataset/model local.
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
- Regression gần nhất: `15 passed, 3 warnings in 1.54s`.
