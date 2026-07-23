# Role 04 — ML B (Collaborative Filtering + Pipeline)

**Owner:** 18- Thanh Loan
**Owns:** `src/recommender_cf.py`, `src/evaluation.py`, `scripts/build_cf_artifacts.py`, `scripts/run_evaluation.py`, `notebooks/04_cf_experiments.ipynb`, CF artifacts trong `artifacts/`

## Trách nhiệm

- Item-based CF trên sparse utility matrix — **recommend phải xong D7 30/07**
- Recommend theo `userId`, loại phim đã rating
- Evaluation HR@10 / NDCG@10 — **deadline D9 01/08** (app freeze)

## ⚠️ Quan trọng

Timeline cũ cho Loan 7 ngày (tuần 2). Timeline mới compress còn **6 ngày (D4–D9)** để app xong 01/08. Phải chạy CF utility + similarity + recommend nhanh hơn. Cụ thể:
- T06 bắt đầu **D4 CN 27/07** (cùng lúc Tâm hand-off parquet)
- T07+T08 dồn vào **D6 T3 29/07**
- T08 phải hand-off cho Tân Dư wire tab cuối **D7 T4 30/07**

## Phase 1 — App Build (D1–D9)

### D1 T5 24/07 — Prep
- [ ] Có mặt họp kickoff
- [ ] Đọc `src/recommender_cf.py` (signature đã có, fix P0/P1 đã apply)
- [ ] Đọc `src/evaluation.py` (đã có `run_evaluation` + `prepare_eval`/`evaluate` split)
- [ ] Đọc `scripts/build_cf_artifacts.py` + `scripts/run_evaluation.py` (stub)

### D4–D5 CN–T2 27–28/07 — T06 Sparse utility matrix **(START EARLY)**

**Mục tiêu:** Artifact utility matrix lưu được cuối D5.

**Sub-task checklist:**
- [ ] Implement `build_utility_matrix(ratings)`:
  - dùng `userId.astype("category")` + `movieId.astype("category")` để map codes
  - build `csr_matrix((data, (rows, cols)))`
  - trả về `(utility, user_ids, movie_ids, user_to_row, movie_to_col)`
- [ ] Implement phần 1 của `scripts/build_cf_artifacts.py::main()`:
  - load `ratings_clean.parquet`
  - build utility → save `utility_matrix.npz` + `movie_id_maps.pkl`
  - in shape + density

**File cần implement:**
- `src/recommender_cf.py` — `build_utility_matrix`
- `scripts/build_cf_artifacts.py` — `main()` (phần T06)

**Done khi:** Artifact utility matrix lưu được, có thể load lại.

### D6 T3 29/07 — T07 Item similarity + T08 Recommend (START)

**T07 — Item similarity:**
- [ ] Implement `build_item_similarity(utility)`:
  - `cosine_similarity(utility.T.tocsr(), dense_output=False)`
- [ ] Mở rộng `main()`: build item_sim → save `item_similarity.npz`
- [ ] In shape + density

**T08 — Recommend (start):**
- [ ] Verify `recommend_for_user(model, movies, user_id, top_k, min_rating=4.0)`
- [ ] Build sparse `liked_mask` (1 × n_movies)
- [ ] Sparse matmul: `scores = (liked_mask @ item_similarity).toarray().ravel()`
- [ ] Mask seen = `-inf`, filter unseen > 0
- [ ] Raise `KeyError`/`ValueError` theo contract

**File:**
- `src/recommender_cf.py` — `build_item_similarity` + verify `recommend_for_user`

### D7 T4 30/07 — T08 Recommend (finish)

**Sub-task checklist:**
- [ ] Chạy `pytest tests/test_recommender.py::test_cf_*` → 4 tests xanh
- [ ] Trên data thật: user 1 trả top-K không chứa phim đã rating
- [ ] Trên data thật: user 999999 raise KeyError
- [ ] Trên data thật: user chỉ rate < 4.0 → raise ValueError
- [ ] Hand-off cho Tân Dư wire tab CF

**Done khi:** Tân Dư gọi được `recommend_for_user` từ `app.py` tab 3.

### D8–D9 T5–T6 31/07–01/08 — T10 Eval HR@10 / NDCG@10

**Sub-task checklist:**
- [ ] Implement `scripts/run_evaluation.py::parse_args()` + `main()`:
  - load processed
  - gọi `run_evaluation(ratings, movies, sample_size=200)`
  - in table + save `reports/cf_eval_scores.csv`
- [ ] Chạy `python scripts/run_evaluation.py`
- [ ] Điền số liệu vào `reports/cf_evaluation.md`
- [ ] Điền `notebooks/04_cf_experiments.ipynb` với sweep (sample size, min_rating)

**File cần implement:**
- `scripts/run_evaluation.py` — `parse_args` + `main`
- `reports/cf_evaluation.md` — điền HR@10/NDCG@10 thật
- `notebooks/04_cf_experiments.ipynb` — sweep results

**Done khi:** Bảng metric trong report, có thể đưa vào final_report section 5.

## Phase 2 — Report & Ship (D10–D14)

### D10 T7 02/08 — T12 Final report (section CF + Eval)

**Sub-task checklist:**
- [ ] Section 5 (CF) trong `reports/final_report.md`
- [ ] Section 5.1 (Eval HR@10/NDCG@10) với bảng số liệu thật
- [ ] Điền: sparse approach, artifact size, sample size, metric table

**File:**
- `reports/final_report.md` — section 5

### D11–D12 CN–T2 03–04/08 — T13 Charts + slides

**Sub-task checklist:**
- [ ] Phối Kiên vẽ chart trong `notebooks/05_evaluation_analysis.ipynb`
- [ ] Slide cho CF + eval (1–2 slide)

## Lưu ý

- Không tạo dense pivot
- Nếu RAM yếu: filter user≥20, movie≥50 + sample users
- `load_npz(path)` chỉ nhận 1 arg
- **App freeze 01/08**: sau D9 chỉ làm report, không sửa logic CF

## ⚠️ Memory planning (rủi ro P1-3)

Trên **MovieLens 25M** (62k movies × 162k users), ma trận item-item similarity
(62k × 62k) có thể chiếm **15–20 GB** ở dạng sparse nếu density > 30%. Máy
16 GB sẽ OOM. Mitigation theo thứ tự ưu tiên:

1. **Pre-filter mạnh hơn trước `build_utility_matrix`**: `min_user=50, min_movie=200`
   (thay vì 20/50) — drops long-tail items tạo noise high-density.
2. **Threshold top-K neighbors per item**: sau khi `cosine_similarity`,
   `argpartition` mỗi hàng lấy top 50 neighbors, zero phần còn lại → bounds
   nnz = O(n_movies × 50).
3. **Sample users**: random 50k users cho utility matrix. CF vẫn work, ít coverage.
4. **Chunk computation**: split N item mỗi lần, `scipy.sparse.hstack` kết quả.

Code CF hiện tại chỉ giữ form `cosine_similarity(utility.T)` đúng theo spec
pseudocode (§13.4). **Loan phải apply mitigation #1 và #2 khi build artifact
T07** nếu gặp OOM. Đã ghi chú trong `src/recommender_cf.py::build_item_similarity`.

## Done khi (toàn sprint)

Tab CF chạy trước 30/07. Eval report có số liệu HR@10/NDCG@10 trước 01/08. Final report section 5 xong D11.
