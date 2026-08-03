# Test cases — owned by QA (Hoàng Đức Kiên)

Status legend: `pending` | `pass` | `fail` | `skip`

Mapping implementation:
- S1, S2, S3 → `tests/test_recommender.py`
- C3, C4 → `tests/test_recommender.py` (C1/C2 = real-data manual)
- F1, F2, F3 → `tests/test_recommender.py`
- D1, D2, D3 → `tests/test_data_pipeline.py`
- A1, A2 → `tests/test_app_smoke.py`

## Simple Recommender

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| S1 | top_k=10 | đúng 10 rows | test_recommender.py::test_simple_top_k | pass |
| S2 | phim ít rating điểm cao | không đứng top nếu num_ratings thấp | test_recommender.py::test_simple_rare_high_rating_not_dominate | pass |
| S3 | columns schema | có movieId, title, genres, avg_rating, num_ratings, weighted_rating | test_recommender.py::test_simple_columns_schema | pass |

## Content-based

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| C1 | Toy Story (1995) | nhiều Animation/Children/Comedy/Fantasy | manual / notebook 03 | pending |
| C2 | Heat (1995) | Action/Crime/Thriller overlap | manual / notebook 03 | pending |
| C3 | title không tồn tại | ValueError / message rõ | test_recommender.py::test_content_missing_title | pass |
| C4 | top-K | không chứa chính input movie | test_recommender.py::test_content_excludes_self | pass |

## Collaborative Filtering

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| F1 | userId tồn tại | top-K, không gồm phim đã rating | test_recommender.py::test_cf_user_in_scope_excludes_seen | pass |
| F2 | userId không tồn tại | KeyError → app fallback Simple | test_recommender.py::test_cf_unknown_user_raises | pass |
| F3 | sparse pipeline | không OOM trên sample | test_recommender.py::test_cf_sparse_pipeline_no_oom | pass |

## Data Pipeline

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| D1 | movies_clean schema | có 6 cột: movieId, title, year, genres, genres_list, genres_text | test_data_pipeline.py::test_d1_movies_schema | pass |
| D2 | year extracted | >=95% phim có year hợp lệ | test_data_pipeline.py::test_d2_movies_year_extracted | pass |
| D3 | ratings fixture + dtype | dtype hợp lệ, rating trong range, no dup, movieId khớp movies | test_data_pipeline.py::test_d3_ratings_filtered_and_dtype | pass |

## App / UX

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| A1 | model adapter predict | load fixture data/model và trả top-K đúng schema | test_app_smoke.py::test_a1_model_adapter_predicts_from_fixture | pass |
| A2 | Streamlit app import | app hiện tại import được và có các hàm render chính | test_app_smoke.py::test_a2_streamlit_app_imports_current_layout | pass |

## Evaluation

| ID | Case | Expected | File | Status |
|----|------|----------|------|--------|
| E1 | HR@10 > 0 | CF tốt hơn random | scripts/run_evaluation.py | pending (until T19) |
| E2 | NDCG@10 > 0 | có ranking chất lượng | scripts/run_evaluation.py | pending (until T19) |
