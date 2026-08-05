# Test Report

Owner: QA (Hoàng Đức Kiên) · Status: updated · Last sync: 2026-08-03

## Summary

Mirror của `tests/test_cases.md`. Cập nhật sau khi chạy:

```bash
.venv313/bin/python -m pytest tests -v
```

| Suite | Tests in code | Cases in `test_cases.md` | Pass | Fail | Skip |
|-------|---------------|--------------------------|------|------|------|
| Recommender core | 14 | S/C/F + metric helpers | 14 | 0 | 0 |
| Data pipeline (D) | 3 | 3 | 3 | 0 | 0 |
| App smoke (A) | 2 | 2 | 2 | 0 | 0 |
| Hybrid model adapter | 3 | adapter contract/fallback | 3 | 0 | 0 |
| ML recommenders | 5 | CF/content/hybrid helpers | 5 | 0 | 0 |
| Edge guards | 19 | regression/error guards | 19 | 0 | 0 |
| **Total** | **46** | **46 + 2 manual** | **46** | **0** | **0** |

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
- Regression gần nhất: `46 passed, 3 warnings in 2.21s`.
