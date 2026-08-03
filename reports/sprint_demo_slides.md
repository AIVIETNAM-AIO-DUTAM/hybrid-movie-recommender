---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Movie Recommendation System · Sprint Demo'
footer: '03/08/2026 · Team 5'
style: |
  section { font-size: 22px; }
  h1 { color: #2C5F8D; }
  h2 { color: #2C5F8D; border-bottom: 2px solid #E5E5E5; padding-bottom: 6px; }
  table { font-size: 16px; }
  img { max-height: 380px; }
---

<!-- _class: lead -->

# Movie Recommendation System

### 3 phương pháp gợi ý phim trên MovieLens 25M

**Team 5 · Sprint Demo · 06/08/2026**

Tân Dư (Lead) · Trần Hoàng Minh Tâm · tran Duong · 18- Thanh Loan · Hoàng Đức Kiên

---

# Vấn đề & mục tiêu

**3 câu hỏi hệ thống trả lời:**

1. **User mới** (chưa có lịch sử) → top-K phim phổ biến
2. **User có phim đầu vào** → phim tương tự theo thể loại
3. **User có lịch sử rating** → gợi ý cá nhân hóa

**Mục tiêu:**
- Pipeline xử lý 25M ratings không OOM
- 3 phương pháp song song + demo Streamlit 3 tab
- Đánh giá định lượng: weighted_rating, genre overlap@K, HR@10/NDCG@10

**Out of scope:** Hybrid production, Neural CF, real-time serving.

---

# Dataset — MovieLens 25M

| | Trước filter | Sau filter (CF) | % giữ lại |
|---|---|---|---|
| Số phim | 59.047 | 13.172 | 22.3% |
| Số user | 162.541 | 162.242 | 99.8% |
| Số rating | 25.000.095 | 24.639.412 | 99.56% |
| Sparsity | — | 98.85% | — |
| Rating mean (C) | — | 3.540 | — |

Pipeline: `clean_movies` (tách year + genres_text) → `clean_ratings` (iterative filter user≥20 / movie≥50).

---

# Kiến trúc hệ thống

```
data/raw/{movies,ratings}.csv
        ↓ src/data_processing.run_pipeline()
data/processed/{movies_clean,ratings_cf,ratings_content}.parquet
        ↓
  ┌─────────────┬─────────────────┬───────────────────┐
  │   Simple    │     Content     │        CF         │
  │ WR = vR+vC  │  CountVec +     │  Item-item cosine │
  │  v+m   v+m  │  cosine sim     │  (sparse, top-K)  │
  └──────┬──────┴────────┬────────┴─────────┬─────────┘
         │               │                  │
         └───────────────┼──────────────────┘
                         ↓
              src/app.py · Streamlit 3 tab
                         ↓
            evaluation/{cf_eval,model_comparison}.csv
```

---

# Simple Recommender

**Công thức IMDb:** `WR = (v/(v+m))·R + (m/(v+m))·C`

- `C = 3.540` (mean), `m = 1.767` (quantile 0.80)
- Filter `num_ratings >= m` trước khi rank

![bg right:50% w:500](../reports/charts/simple_top10.png)

Top-3:
1. **Shawshank Redemption** (4.40, 81k ratings)
2. **Godfather** (4.30, 52k)
3. **Usual Suspects** (4.26, 55k)

Bias: all top-10 là Crime/Drama — hạn chế của IMDb formula.

---

# Content-based Recommender

**Phương pháp:** `CountVectorizer(token_pattern=r"\S+")` + cosine similarity (sparse, on-demand).

**Kết quả Genre overlap@10** (tỷ lệ top-K chia sẻ ≥1 genre với input):

| Input | Overlap@10 |
|---|---|
| Toy Story (1995) | 1.00 |
| Heat (1995) | 1.00 |
| Pulp Fiction (1994) | 1.00 |
| Matrix, The (1999) | 1.00 |
| Inception (2010) | 1.00 |

> Cosine = 1.0 cho mọi phim cùng genre → metric bão hòa, cần thêm metadata (director, actors, plot) để phân biệt trong cùng nhóm.

---

# Collaborative Filtering (Item-based)

**Pipeline:**
1. Utility matrix (sparse CSR): user × movie = 162.242 × 13.172, nnz = 24.6M
2. Item-item cosine, **top-100 neighbors/item** (chống OOM, spec §11)
3. Score = `(liked_mask @ item_similarity)` (sparse matmul)

**Artifacts:**

| File | Shape | Size |
|---|---|---|
| `utility_matrix.npz` | (162.242, 13.172) | 51 MB |
| `item_similarity.npz` | (13.172, 13.172) | 6.9 MB |
| `movie_id_maps.pkl` | — | 2.0 MB |

Cold-start: user mới → fallback Simple.

---

# Đánh giá CF — HR@10 / NDCG@10

Leave-last-out by timestamp, sample 200 user:

| Metric | Value |
|---|---|
| HR@10 | 0.02 |
| NDCG@10 | 0.0141 |

![bg right:50% w:480](../reports/charts/hr_vs_topk.png)

**Tại sao thấp?**
- Item-based CF trên long-tail users
- Chưa tune `min_rating` và neighbor `top_k`
- Hold-out là 1 phim/user → baseline khó

Cải thiện: tăng top_k, giảm min_rating (4.0 → 3.5), thử user-based.

---

# So sánh 3 phương pháp + Hybrid

![w:800](../reports/charts/model_comparison.png)

Hybrid α=0.8 thắng precision@10 (0.076) và ngang CF về hit_rate (0.44).
Content đơn lẻ quá yếu (precision 0.003) do genre cosine bão hòa.

---

# Hybrid — alpha sweep

![w:720](../reports/charts/hybrid_alpha_sweep.png)

Đỉnh precision/recall tại **α = 0.8** (CF weight cao).
Rơi mạnh khi α < 0.6 — Content signal không đủ mạnh để đóng góp.

---

# Per-user recall distribution

![w:720](../reports/charts/per_user_recall_distribution.png)

**Bimodal:** ~46% user có hit, ~54% không. CF không cover được long-tail users
— gap để cải thiện cho sprint sau (matrix factorization, hoặc personalization fallback).

---

# Test suite & QA

**32 tests pass 100%** (17 → 32 trong sprint, +15 edge-guard mới)

| Suite | Tests | Pass |
|---|---|---|
| Simple / Content / CF | 13 | 13 |
| Eval helpers | 1 | 1 |
| Data pipeline (D) | 3 | 3 |
| App smoke (A) | 2 | 2 |
| Edge guards | 13 | 13 |

**Bug đã fix:** B1 (3-tuple unpack), B2 (ratings alias), B3 (genre overlap), B4 (OOM top-K sparsify).

---

# Demo Streamlit app

3 tab (chạy: `streamlit run src/app.py`):

1. **Simple** — chọn genre (optional) → top-K weighted rating
2. **Content** — gõ title → 10 phim tương tự theo genre
3. **CF** — gõ `userId` → top-K personalized (cold-start fallback Simple)

Catches: missing parquet, unknown user, no liked items, no candidates → đều warn sạch,
không crash.

---

# Bài học & kế hoạch

**Bài học sprint này:**
1. **LFS setup từ đầu** — 3 fixture parquet gây rebase conflict (binary ↔ pointer)
2. **Edge-case test song song code** — không phải để cuối sprint
3. **Dành 1 ngày riêng cho eval sweep** — HR@10 thấp vì không kịp tune
4. **Deterministic sort keys** — `weighted_rating desc, num_ratings desc, movieId asc`

**Sprint sau (T14):**
- Rehearse demo (05/08)
- Bổ sung charts còn thiếu nếu review yêu cầu
- Ship ngày 06/08

---

<!-- _class: lead -->

# Cảm ơn!

### Q&A

Repo: `fix/sprint-gaps-d8` · Reports: `reports/final_report.md`
Charts: `reports/charts/` (chạy `jupyter nbconvert --execute notebooks/05_evaluation_analysis.ipynb`) · Eval: `evaluation/`
