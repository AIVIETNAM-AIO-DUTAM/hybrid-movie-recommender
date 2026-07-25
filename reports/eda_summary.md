# EDA Summary

Owner: Data Engineer · Status: edited

## Raw

| File | Rows | Notes |
|------|------|-------|
| movies.csv | 62,423 | |
| ratings.csv | 25,000,095 | ~678MB |



## After clean

| File | Rows | Filters |
|------|------|---------|
| movies_clean.parquet |  62,423 | |
| ratings_cf.parquet | 24,639,412 | user≥20, movie≥50 |
| ratings_content.parquet | 24,945,390 | user≥20, movie≥5 |

## Key findings
- Sparsity:  0.9973951609474271 / ~0.995307 (cd)/ ~0.98847(content)
- Rating mean (C):3.5338545
- Genre notes: 19 (giữ nguyên sau cleaned)

