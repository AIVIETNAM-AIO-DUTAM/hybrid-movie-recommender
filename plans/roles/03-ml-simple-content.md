# Role 03 — ML A (Simple + Content-based)

**Owner:** tran Duong
**Owns:** `src/recommender_simple.py`, `src/recommender_content.py`, `notebooks/03_modeling.ipynb`

## Trách nhiệm

- Simple: IMDb-style weighted rating + top-K (+ optional genre filter) — **deadline 28/07**
- Content-based: genre vector + cosine similarity — **deadline 28/07**
- Genre overlap@K metric cho final report (Phase 2)

## Phase 1 — App Build (D1–D9)

### D4–D5 CN–T2 27–28/07 — T04 Simple Recommender

**Mục tiêu:** `recommend_top_movies(top_k=10)` trả đúng 10 phim theo weighted_rating. Deadline **D5 28/07** để Tân Dư wire tab.

**Sub-task checklist:**
- [ ] Implement `build_movie_stats(ratings, m_quantile=0.80)`
- [ ] Implement `recommend_top_movies(movies, stats, top_k=10, genre=None)`
- [ ] Filter `num_ratings >= m` (fallback all nếu rỗng)
- [ ] Sort deterministic: WR desc → num_ratings desc → movieId asc
- [ ] Chạy `pytest tests/test_recommender.py::test_simple_*` → xanh
- [ ] Trên parquet thật: verify top-10 (Shawshank/Pulp Fiction)

**Công thức (PDF §3.1):**

```python
WR = (v / (v + m)) * R + (m / (v + m)) * C
```

**File cần implement:**
- `src/recommender_simple.py` — điền TODO

### D4–D5 CN–T2 27–28/07 — T05 Content-based (parallel với T04)

**Mục tiêu:** `recommend_similar_movies("Toy Story (1995)", top_k=10)` trả phim genre overlap.

**Sub-task checklist:**
- [ ] Implement `build_content_model(movies)` với `token_pattern=r"\S+"` (giữ "Sci-Fi")
- [ ] Implement `recommend_similar_movies(model, movie, top_k=10)`
- [ ] Resolve idx (exact → prefer-year partial)
- [ ] Guard zero-norm query (no genres) → `ValueError`
- [ ] Cosine on-demand: `cosine_similarity(query_row, all)`
- [ ] Exclude self + **argsort(-scores) KHÔNG [::-1]**
- [ ] Tính `shared_genres`
- [ ] Chạy `pytest tests/test_recommender.py::test_content_*` → xanh
- [ ] Trên parquet thật: verify Toy Story + Heat

**File cần implement:**
- `src/recommender_content.py` — điền TODO

**Done khi (T04+T05):** Tân Dư gọi được 2 hàm từ `app.py` tab 1–2 trước D6.

### D6–D9 29/07–01/08 — Buffer + support

- [ ] Hỗ trợ Tân Dư wire tab nếu contract sai
- [ ] Hỗ trợ Kiên test nếu fixture fail

## Phase 2 — Report & Ship (D10–D14)

### D10 T7 02/08 — T12 Final report (section Simple + Content)

**Sub-task checklist:**
- [ ] Section 3 (Simple Recommender) trong `reports/final_report.md`
- [ ] Section 4 (Content-based) trong `reports/final_report.md`
- [ ] Điền: công thức, code snippet, ví dụ top-10 + Toy Story similar

**File:**
- `reports/final_report.md` — section 3, 4

### D10–D11 T7–CN 02–03/08 — Genre overlap@K metric

**Mục tiêu:** Metric đánh giá chất lượng content (PDF §6.2).

**Sub-task checklist:**
- [ ] Implement `genre_overlap_at_k(recommendations, input_genres, k=10) -> float` trong `recommender_content.py`
- [ ] Tính cho 5 phim: Toy Story, Heat, Pulp Fiction, The Matrix, Inception
- [ ] Ghi bảng vào `notebooks/03_modeling.ipynb` và `reports/final_report.md` section 4

**Files:**
- `src/recommender_content.py` — thêm `genre_overlap_at_k`
- `notebooks/03_modeling.ipynb` — bảng metric

### D13–D14 — Polish + rehearsal
- [ ] Cross-review report
- [ ] Có mặt rehearsal D14

## Không làm

- Không sửa `recommender_cf.py` (của Loan)
- Không sửa `data_processing.py` (của Tâm)

## Done khi (toàn sprint)

Tab 1 (Simple) + tab 2 (Content) chạy trước 28/07. Final report section 3, 4 + overlap@K xong D11 03/08.
