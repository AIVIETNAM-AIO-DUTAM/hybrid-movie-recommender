# Data Dictionary (edited)

Cập nhật bởi **Data Engineer**.

## movies.csv (raw)

| Column | Type | Meaning | Notes |
|--------|------|---------|-------|
| movieId | int | ID phim | Unique |
| title | string | Tên + năm | VD: `Toy Story (1995)` |
| genres | string | Thể loại `|` | Có thể `(no genres listed)` |

## ratings.csv (raw)

| Column | Type | Meaning | Notes |
|--------|------|---------|-------|
| userId | int | ID user | |
| movieId | int | ID phim | FK → movies |
| rating | float | 0.5–5.0 | Step 0.5 |
| timestamp | int | Unix time | Dùng split train/test |

## movies_clean.parquet (target)

| Column | Meaning |
|--------|---------|
| movieId | ID |
| title | Tên sạch |
| year | Năm tách từ title |
| genres | Chuỗi gốc |
| genres_list | list[str] |
| genres_text | space-separated cho vectorizer |

## ratings_clean.parquet và ratings_cf.parquet (target)

| Column | Meaning |
|--------|---------|
| userId | int32 |
| movieId | int32 |
| rating | float32 |
| timestamp | int64 |

### Filter 
ratings_cf.parquet
- Giữ user ≥ 20 ratings
- Giữ movie ≥ 50 ratings
ratings_content.parquet
- Giữ user ≥ 20 ratings
- Giữ movie ≥ 5 ratings

Ghi lại số row trước/sau filter vào `reports/eda_summary.md`.
