# Final Report — Movie Recommendation System

Owner: Tech Lead + cả nhóm · Due: T4 06/08/2026 23:59

> **Hướng dẫn điền:** mỗi section có tên owner phụ trách. Owner điền nội dung,
> giữ heading. Reference spec: `document.pdf` mục tương ứng.

## 1. Giới thiệu & mục tiêu

Owner: **Tân Dư**

Hệ thống gợi ý phim (Movie Recommendation System) được xây dựng trên dataset **MovieLens 25M**, đáp ứng 3 câu hỏi cốt lõi của người dùng cuối theo spec (`document.pdf` mục 1):

1. **User mới** (chưa có lịch sử rating): xem top-K phim phổ biến nhất → tab **Simple**.
2. **User có phim đầu vào**: tìm phim tương tự theo thể loại → tab **Content**.
3. **User đã có lịch sử rating**: gợi ý cá nhân hóa theo `userId` → tab **CF**.

**3 mục tiêu của project:**

- Xây dựng pipeline dữ liệu end-to-end từ raw CSV → parquet, xử lý 25M ratings mà không OOM.
- Tr triển khai 3 phương pháp gợi ý song song (Simple, Content, CF) trên cùng data layer, với Streamlit demo 3 tab.
- Đánh giá khách quan bằng metric định lượng (weighted rating, genre overlap@K, HR@10/NDCG@10).

**Scope 3 phương pháp (PDF mục 3):** Simple (IMDb weighted rating), Content-based (genre cosine), Item-based Collaborative Filtering.

**Out of scope (PDF mục 10 — Won't-have):** Hybrid recommender, Neural CF / deep learning, metadata ngoài genre, real-time serving infrastructure.

## 2. Dataset & preprocessing

Owner: **Trần Hoàng Minh Tâm**

- MovieLens 25M: `movies.csv` (62.423 phim, 2.9MB) + `ratings.csv` (25.000.095 ratings, ~678MB)
- Schema (PDF mục 2.1, 2.2) — tham chiếu `docs/data-dictionary.md`
- Pipeline clean (`src/data_processing.py`):
  - Tách `year` từ `title` bằng regex `\((\d{4})\)[^()]*$`
  - Tách `genres` thành `genres_list` + `genres_text` (lọc sentinel `(no genres listed)`)
  - Lọc iterative: `user ≥ 20 ratings` AND `movie ≥ 50 ratings` (CF) / `movie ≥ 5 ratings` (content)
- Bảng số liệu trước/sau filter — copy từ `reports/eda_summary.md`

| | Trước filter | Sau filter | % giữ lại |
|---|---|---|---|
| Số phim | 59.047 | 32.711 (content) - 13.172 (cf) | ~55.4% & ~22.3% |
| Số user | 162.541 | 162.516 (content) - 162.242 (cf) | ~99.98 & ~ 99.8%|
| Số rating | 25.000.095 | 24.945.390 (content) - 24.639.412 (cf) | ~99.78% & ~99.56% |
| Sparsity | ~99.74% | ~99.53% (content) - ~98.847 (cf) | — |
| Rating mean (C) | ~3.534 | ~3.535 (content) - ~3.54 (cf) | — |

Nguồn: `reports/eda_summary.md` (sau `src/data_processing.py:run_pipeline()`). Sparsity CF = 0.01153 (utility matrix), C = 3.5403, m (quantile 0.80) = 1.767 ratings.

## 3. Simple Recommender

Owner: **tran Duong**

- Mục tiêu (PDF mục 3.1): top-K phổ biến cho user mới
- Phương pháp: IMDb weighted rating (`src/recommender_simple.py`)
  - Công thức: `WR = (v/(v+m))*R + (m/(v+m))*C`
  - `C = 3.5403` (mean toàn dataset), `m = 1.767` (quantile 0.80 của `num_ratings`)
  - Filter `num_ratings >= m` trước khi rank
  - Sort: `weighted_rating desc, num_ratings desc, movieId asc` (đảm bảo deterministic)
- Demo: tab Simple trong Streamlit (`src/app.py`)
- Ưu điểm / hạn chế (PDF mục 3.1):
  - ✅ Ưu điểm: nhanh, không cold-start, giải thích dễ
  - ❌ Hạn chế: không cá nhân hóa, bias phim phổ biến

Bảng top-10 phim (copy từ app, computed trên `ratings_cf.parquet`):

| movieId | title | genres | avg_rating | num_ratings | weighted_rating |
|---|---|---|---|---|---|
| 318 | Shawshank Redemption, The (1994) | Crime, Drama | 4.4138 | 81.418 | 4.3953 |
| 858 | Godfather, The (1972) | Crime, Drama | 4.3246 | 52.463 | 4.2991 |
| 50 | Usual Suspects, The (1995) | Crime, Mystery, Thriller | 4.2848 | 55.344 | 4.2618 |
| 527 | Schindler's List (1993) | Drama, War | 4.2479 | 60.371 | 4.2277 |
| 1221 | Godfather: Part II, The (1974) | Crime, Drama | 4.2621 | 34.171 | 4.2266 |
| 2959 | Fight Club (1999) | Action, Crime, Drama, Thriller | 4.2284 | 58.737 | 4.2083 |
| 1193 | One Flew Over the Cuckoo's Nest (1975) | Drama | 4.2188 | 36.044 | 4.1871 |
| 904 | Rear Window (1954) | Mystery, Thriller | 4.2381 | 20.157 | 4.1819 |
| 1203 | 12 Angry Men (1957) | Drama | 4.2429 | 16.563 | 4.1752 |
| 296 | Pulp Fiction (1994) | Comedy, Crime, Drama, Thriller | 4.1889 | 79.649 | 4.1749 |

## 4. Content-based Recommender

Owner: **tran Duong**

- Mục tiêu (PDF mục 3.2): phim tương tự theo genre
- Phương pháp (`src/recommender_content.py`): `CountVectorizer(token_pattern=r"\S+")` + cosine similarity (sparse, on-demand)
  - Token pattern `\S+` giữ nguyên các genre có dấu gạch như "Sci-Fi"
  - Similarity tính tại query-time thay vì pre-materialize → tránh 30GB dense matrix cho 62k movies
- Demo: tab Content với "Toy Story (1995)", "Heat (1995)"
- Genre overlap@K metric (PDF mục 6.2) — `genre_overlap_at_k()` đếm tỷ lệ top-K recommendations chia sẻ ≥1 genre với input

| Input | Genre overlap@10 |
|-------|------------------|
| Toy Story (1995) | 1.00 |
| Heat (1995) | 1.00 |
| Pulp Fiction (1994) | 1.00 |
| Matrix, The (1999) | 1.00 |
| Inception (2010) | 1.00 |

Nguồn: `genre_overlap_at_k()` trên `movies_clean.parquet` (D8). Lưu ý: title trong DB là `Matrix, The (1999)` (MovieLens convention).

## 5. Collaborative Filtering

Owner: **18- Thanh Loan**

- Mục tiêu (PDF mục 3.3): cá nhân hóa theo userId
- Phương pháp: Item-based CF, sparse `csr_matrix`, cosine item-item (`src/recommender_cf.py`)
- Pipeline (PDF mục 4):
  - load → clean → sparse utility → item similarity → recommend
  - `build_utility_matrix()`: dùng pandas `category` codes → fill `csr_matrix` (user × movie)
  - `build_item_similarity()`: cosine theo từng chunk + giữ top-K neighbors/item (spec §11 OOM mitigation)
- Cold-start fallback (spec §11): user mới → Simple
- Demo: tab CF với userId=1, userId=42

**Mô tả CF artifacts** (sau `scripts/build_hybrid_artifacts.py`):

| Artifact | Shape | NNZ | Density | Size |
|---|---|---|---|---|
| `utility_matrix.npz` | (162.242, 13.172) | 24.639.412 | 0.01153 | 51.0 MB |
| `item_similarity.npz` | (13.172, 13.172) | 1.317.200 | 0.00759 | 6.9 MB |
| `movie_id_maps.pkl` | — | — | — | 2.0 MB |

Score cho mỗi candidate movie = `(liked_mask @ item_similarity)` (sparse matmul, 1×n_liked · n_liked×n_movies). Loại trừ phim đã xem (set score = -inf). Bỏ candidates có similarity = 0 để tránh noise.

### 5.1 Evaluation

| Metric | Value | Sample size | Notes |
|--------|-------|-------------|-------|
| HR@10 | 0.02 | 200 | leave-last-out by timestamp; evaluated users that CF could score |
| NDCG@10 | 0.0141 | 200 | same sample |
| HR@10_all | 0.02 | 200 | includes cold-start / no-candidate users in denominator |
| NDCG@10_all | 0.0141 | 200 | same |

Nguồn: `reports/cf_evaluation.md` / `evaluation/cf_eval_scores.csv`.

> HR@10 = 0.02 thấp — đặc trưng cho item-based CF leave-last-out trên MovieLens 25M khi chưa tune `min_rating` / neighbor top-K. Đường cải thiện: tăng `top_k` của item similarity, giảm `min_rating` để mở rộng pool liked items, hoặc thử user-based CF.

## 6. So sánh 3 phương pháp

Owner: **Tân Dư**

| Tiêu chí | Simple | Content | CF |
|----------|--------|---------|-----|
| Cá nhân hóa | Không | Theo phim input | Theo user |
| Cần data | ratings | movies | ratings |
| User mới | Tốt | Tốt | Fallback Simple |
| Tốc độ | Rất nhanh | Nhanh | Phụ thuộc sparse |
| Khả năng giải thích | Dễ | Dễ (genre) | Trung bình |
| Metric | weighted_rating | genre overlap@K | HR@10, NDCG@10 |

Tham chiếu PDF mục 3.4.

## 7. Demo & limitation

Owner: **Tân Dư**

- **Official demo**: `src/app.py` (3 tab Streamlit — Simple / Content / CF). Có behavioral test (`tests/test_app_smoke.py` A1+A2) và là path được cite trong README, slides, DoD.
- **Parallel UI**: `src/app/streamlit_app.py` (MicroLens UI, Hybrid α-blend). Không có behavioral test (chỉ AST parse guard), phụ thuộc `model/knn_cf/` + `model/knn_content/` artifacts sinh bởi `scripts/build_hybrid_artifacts.py`. Đánh giá offline qua `scripts/run_hybrid_evaluation.py`.
- **CF tab trong `src/app.py`**: status legacy. Tab vẫn render nhưng `scripts/build_hybrid_artifacts.py` chỉ sinh artifacts cho `src/app/streamlit_app.py` (đường `model/knn_cf/`), không sinh `artifacts/utility_matrix.npz` cho `src/app.py`. → Tab CF trong `src/app.py` gần như luôn fallback về Simple Recommender. Đã ghi nhận là hạn chế sprint này, sẽ unify 2 app sprint sau.
- Hạn chế (PDF mục 11 rủi ro):
  - Content chỉ có genre → gợi ý thô (cosine = 1.0 cho mọi phim cùng genre)
  - CF cần user có rating → cold-start
  - HR@10 còn thấp → chưa tune hyperparameter
  - Không có Neural CF (out of scope)

## 8. Test & Evaluation

Owner: **Hoàng Đức Kiên**

Chiến lược test là unit test (pytest) + manual smoke test trên Streamlit app demo. Tổng cộng có **32 test cases pass 100%** (trước đây 17, sprint này đã bổ sung 15 test edge-guard mới trong `tests/test_edge_guards.py`; tổng = 17 + 15 = 32).

| Suite | Tests | Pass | Skip | Notes |
|---|---|---|---|---|
| Simple (S) | 3 | 3 | 0 | top-K + schema + rare-rating guard |
| Content (C) | 5 | 5 | 0 | excl self, missing title, genre overlap@K (all/none share) |
| CF (F) | 5 | 5 | 0 | user in scope, unknown user raises, sparse no-OOM, no-liked raises |
| Eval helpers | 1 | 1 | 0 | HR@10 / NDCG@10 + leave-last-out timestamp tie determinism |
| Data pipeline (D) | 3 | 3 | 0 | movies schema, year extract, ratings filter+dtype |
| App smoke (A) | 2 | 2 | 0 | A1 missing data warns cleanly + A2 three tabs render |
| Edge guards | 15 | 15 | 0 | CF artifacts meta roundtrip, n_ratings drift, empty input raises (×4), orphaned movieIds warns, content self-leak, single-movie empty, leave-last-out determinism, load_processed_missing, load_cf_artifacts_missing_file, streamlit parse guard |
| **Total** | **32** | **32** | **0** | |

**Bug đã fix trong sprint (từ `reports/test_report.md`):**

- B1 (P1): `load_processed()` trả 3-tuple nhưng `src/app.py` unpack 2 → crash demo. Fixed D8.
- B2 (P1): Tests D1–D3 trỏ `ratings_clean.parquet` trong khi pipeline ghi `ratings_cf.parquet`. Fixed bằng alias + test update.
- B3 (P2): `genre_overlap_at_k` thiếu so sánh plan T16 / DoD Content. Fixed D8.
- B4 (P1): Full item-similarity trên 25M dễ OOM. Fixed bằng top-K sparsify + chunk processing.

**Eval CF:** `reports/cf_evaluation.md` + chart từ `notebooks/05_evaluation_analysis.ipynb`. HR@10/NDCG@10 đã có số liệu trên sample 200 user, lưu tại `evaluation/cf_eval_scores.csv`.

## 9. Phân công & retrospect

Owner: **Tân Dư** + cả nhóm

**Ownership table** (copy từ `TEAM_BOARD.md` roster):

| Người | Role |
|-------|------|
| Tân Dư | Tech Leader (app, integration, report) |
| Trần Hoàng Minh Tâm | AI Engineer (Data pipeline, EDA) |
| tran Duong | AI Engineer (Simple + Content models) |
| 18- Thanh Loan | AI Engineer (CF pipeline, eval) |
| Hoàng Đức Kiên | QA / Reviewer (tests, bug list) |

**Timeline thực tế so với plan:**

- Phase 1 (24/07 → 01/08): hoàn thành đúng milestone APP FREEZE — 3 recommender + app demo + eval đều xanh. Trễ nhẹ ở T05 (Content) và T06–T08 (CF) do xử lý OOM (B4) mất 1 ngày.
- Phase 2 (02/08 → 06/08): final report + charts đang chạy đúng tiến độ D11.

**Bài học cho sprint sau:**

1. **LFS setup từ đầu**: 3 file parquet fixture gây rebase conflict (binary vs LFS pointer) → set up `.gitattributes` + `git lfs install` trước khi commit dữ liệu lớn.
2. **Edge-case test sớm**: 13 test edge-guard mới đáng lẽ nên viết song song với model code, không phải đợi đến cuối sprint.
3. **Eval hyperparameter sweep**: HR@10 thấp vì không có thời gian tune → dành buffer 1 ngày riêng cho eval experiments trong sprint sau.
4. **Determinism**: thêm sort key phụ (`num_ratings desc, movieId asc`) vào Simple Recommender đã giải quyết bug unstable output giữa các run — nên là convention mặc định cho mọi recommender.

## Appendix

- `document.pdf` — spec gốc
- `TEAM_BOARD.md` — Kanban
- `plans/roles/` — chi tiết task từng người
- `reports/eda_summary.md` — số liệu EDA
- `reports/test_report.md` — kết quả test
- `reports/cf_evaluation.md` — kết quả eval
- `reports/charts/` — 7 biểu đồ phân tích (HR@10, NDCG@10, model comparison, hybrid alpha sweep, per-user recall, simple top-10, simple top-100 genres)
- `reports/sprint_demo_slides.md` — slides Marp cho demo ngày 06/08
- `notebooks/05_evaluation_analysis.ipynb` — notebook sinh charts
