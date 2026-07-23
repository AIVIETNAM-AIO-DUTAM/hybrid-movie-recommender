# Movie Recommendation System — Sprint Plan

> **Sprint:** 24/07/2026 → 30/07/2026 (7 ngày, 2 phase)
> **Team:** 5 người · **Dataset:** MovieLens 25M
> **Scope:** 3 phương pháp Recommender · **Demo:** T4 30/07

---

## Team Roster

| # | Tên | Role | Focus chính |
|---|-----|------|-------------|
| 🎯 | **Tân Dư** | Tech Leader | Streamlit app, integration, demo |
| 📊 | **Trần Hoàng Minh Tâm** | AI Engineer (Data) | EDA, data pipeline, parquet |
| 🤖 | **tran Duong** | AI Engineer (Model) | Simple + Content-based |
| ⚙️ | **18- Thanh Loan** | AI Engineer (Pipeline) | Collaborative Filtering + Eval |
| 🧪 | **Hoàng Đức Kiên** | QA / Reviewer | Test cases, regression, eval analysis |

---

## Timeline Snapshot

```
        T5        T6        CN       T2        T3        T4
        24/07     25/07     27/07    28/07     29/07     30/07
  ──────┼─────────┼─────────┼────────┼─────────┼─────────┼──────
  P1    │■■■■■■■■■│■■■■■■■■■│▣ HỌP   │         │         │
  Data  │  T10    │         │ T13    │         │         │
  Model │  T11    │  T12    │ T14    │         │         │
  ──────┼─────────┼─────────┼────────┼─────────┼─────────┼──────
  P2    │         │         │        │■■■■■■■■■│■■■■■■■■■│▣ REHEARSE
  Pipe  │         │         │        │  T15    │  T17    │  T19+T21
  Model │         │         │        │  T16    │  T18    │
  QA    │         │         │        │         │         │  T20
  ──────┴─────────┴─────────┴────────┴─────────┴─────────┴──────
```

**Legend:** ■ làm việc · ▣ milestone · P1 = Phase 1 · P2 = Phase 2

### Hai phase

| Phase | Từ | Đến | Mục tiêu | Số task |
|-------|----|----|----------|---------|
| **Phase 1 — Data + Baseline** | T5 24/07 | CN 27/07 | Parquet + Simple + Content + Demo v1 + Test v1 | 5 |
| **Phase 2 — CF + Ship** | T2 28/07 | T4 30/07 | CF artifact + recommend + eval + Demo v2 + Final report | 7 |

**Họp:** T5 (kickoff) + CN (review tuần 1) + T4 (rehearsal)

---

## Task Details

### Phase 1 — Data + Baseline

#### 🟦 T10 · Clean data → parquet + data dictionary

| Field | Value |
|-------|-------|
| **Owner** | Trần Hoàng Minh Tâm |
| **Day** | T5 24/07 (1 ngày) |
| **Depends on** | T00 (setup) |
| **Deliverable** | `movies_clean.parquet`, `ratings_clean.parquet`, `data-dictionary.md`, 2 EDA notebooks |

**Mô tả:**
Làm sạch MovieLens 25M (`movies.csv` 2.9MB + `ratings.csv` 647MB), áp filter user≥20 ratings / movie≥50 ratings, lưu parquet với dtype tối ưu (`int32/int32/float32/int64`). EDA movies + ratings trong notebook để có số liệu cho `reports/eda_summary.md`.

**Acceptance:**
- Parquet tồn tại với schema: `movieId, title, year, genres, genres_list, genres_text` (movies) và `userId, movieId, rating, timestamp` (ratings)
- `python scripts/run_pipeline.py` chạy không lỗi
- ML A (Duong) load được parquet qua `load_processed()`

---

#### 🟦 T11 · Simple Recommender (weighted rating)

| Field | Value |
|-------|-------|
| **Owner** | tran Duong |
| **Day** | T5 24/07 (1 ngày) |
| **Depends on** | T10 (parquet) |
| **Deliverable** | `src/recommender_simple.py` hoàn chỉnh |

**Mô tả:**
Implement IMDb-style weighted rating để gợi ý top-K phim phổ biến cho mọi user (cold-start baseline).

**Công thức (spec §3.1):**
```
WR = (v / (v + m)) * R + (m / (v + m)) * C
v = num_ratings, R = avg_rating
C = mean toàn dataset, m = quantile(0.80)
```

**Acceptance:**
- `recommend_top_movies(top_k=10)` trả đúng 10 rows
- Phim ít rating không chiếm top vô lý (test S2 xanh)
- Trên data thật: top-10 chứa phim kinh điển (Shawshank, Pulp Fiction, ...)

---

#### 🟦 T12 · Content-based Recommender (genre cosine)

| Field | Value |
|-------|-------|
| **Owner** | tran Duong |
| **Day** | T6 25/07 (1 ngày) |
| **Depends on** | T10 |
| **Deliverable** | `src/recommender_content.py` hoàn chỉnh |

**Mô tả:**
Đề xuất phim tương tự theo genre dùng CountVectorizer + cosine similarity (sparse, on-demand để tránh OOM trên 62K phim).

**Acceptance:**
- `recommend_similar_movies("Toy Story (1995)", top_k=10)` trả phim có genre Animation/Children/Comedy overlap
- `recommend_similar_movies("Heat (1995)")` trả phim Action/Crime/Thriller
- Movie không tồn tại → `ValueError` rõ ràng
- Không trả chính phim đầu vào

---

#### 🟦 T13 · Demo Streamlit v1 (Simple + Content)

| Field | Value |
|-------|-------|
| **Owner** | Tân Dư |
| **Day** | CN 27/07 (0.5 ngày) |
| **Depends on** | T11, T12 |
| **Deliverable** | `src/app.py` chạy được 2 tab |

**Mô tả:**
Wire Simple + Content vào Streamlit app, smoke test trên máy demo.

**Acceptance:**
- Tab Simple trả top-K với slider `top_k` và filter genre
- Tab Content trả similar cho "Toy Story (1995)"
- Lỗi `ValueError` hiển thị bằng `st.error`, không crash UI

---

#### 🟦 T14 · Test suite v1 + bug list tuần 1

| Field | Value |
|-------|-------|
| **Owner** | Hoàng Đức Kiên |
| **Day** | CN 27/07 (1 ngày) |
| **Depends on** | T13 |
| **Deliverable** | `tests/`, `reports/test_report.md`, bug log |

**Mô tả:**
Chạy test trên data thật sau Demo v1, log bug, bổ sung test case S3 (columns schema).

**Acceptance:**
- `pytest tests/ -v` chạy với parquet thật, ít nhất 9 tests cũ vẫn xanh
- Thêm test S3 mới + case C1/C2 (Toy Story / Heat genre overlap) trên data thật
- Bug list tuần 1 có ít nhất 1 entry (dù P2 UX)

---

### Phase 2 — CF + Ship

#### 🟧 T15 · Sparse utility matrix

| Field | Value |
|-------|-------|
| **Owner** | 18- Thanh Loan |
| **Day** | T2 28/07 (1 ngày) |
| **Depends on** | T10 |
| **Deliverable** | `artifacts/utility_matrix.npz`, `artifacts/movie_id_maps.pkl` |

**Mô tả:**
Build sparse CSR utility matrix (user × movie) từ `ratings_clean.parquet`, save artifact để CF có thể load nhanh.

**Acceptance:**
- Artifact tồn tại, shape `(n_users, n_movies)`, density < 1%
- `python scripts/build_cf_artifacts.py` chạy không OOM
- `load_cf_artifacts()` trả CFModel đầy đủ

---

#### 🟧 T16 · Content genre overlap@K metric

| Field | Value |
|-------|-------|
| **Owner** | tran Duong |
| **Day** | T2 28/07 (0.5 ngày) |
| **Depends on** | T12 |
| **Deliverable** | `genre_overlap_at_k()` trong `recommender_content.py` |

**Mô tả:**
Metric đánh giá chất lượng content-based: % phim trong top-K có ≥1 genre trùng với phim input (spec §6.2). Tính cho 5 phim tiêu biểu (Toy Story, Heat, Pulp Fiction, Matrix, Inception).

**Acceptance:**
- `genre_overlap_at_k` trả 1.0 khi tất cả top-K cùng genre
- Có bảng metric cho 5 phim → đưa vào final_report

---

#### 🟧 T17 · Item-item similarity artifact

| Field | Value |
|-------|-------|
| **Owner** | 18- Thanh Loan |
| **Day** | T3 29/07 (1 ngày) |
| **Depends on** | T15 |
| **Deliverable** | `artifacts/item_similarity.npz` |

**Mô tả:**
Tính item-item cosine similarity (movie × movie, sparse) trên utility matrix, save artifact.

**Acceptance:**
- Artifact sparse tồn tại, shape `(n_movies, n_movies)`
- `load_cf_artifacts()` trả CFModel đầy đủ (utility + similarity + maps)

---

#### 🟧 T18 · `recommend_for_user()` + exclude seen

| Field | Value |
|-------|-------|
| **Owner** | 18- Thanh Loan |
| **Day** | T3 29/07 (1 ngày) |
| **Depends on** | T17 |
| **Deliverable** | `recommend_for_user()` hoàn chỉnh |

**Mô tả:**
Hàm recommendation cá nhân hóa theo userId, dùng sparse matmul `(1×n) @ (n×n)`, loại phim đã rating, raise `KeyError` (user không tồn tại) hoặc `ValueError` (cold-start) để app fallback Simple.

**Acceptance:**
- User 1 trả top-K, không chứa phim user 1 đã rating
- User 999999 → `KeyError` → app fallback Simple
- User không có phim nào rate ≥ 4.0 → `ValueError` → app fallback Simple

---

#### 🟥 T19 · Eval HR@10 / NDCG@10

| Field | Value |
|-------|-------|
| **Owner** | 18- Thanh Loan + Hoàng Đức Kiên |
| **Day** | T4 30/07 (1 ngày) |
| **Depends on** | T18 |
| **Deliverable** | `reports/cf_evaluation.md`, `notebooks/05_evaluation_analysis.ipynb` |

**Mô tả:**
Leave-last-out split (per-user, latest timestamp → test), build CF trên train, sample 200 user, recommend top-10, tính HR@10/NDCG@10. QA spot-check + vẽ chart.

**Acceptance:**
- `python scripts/run_evaluation.py` chạy không crash
- Bảng HR@10/NDCG@10 có số liệu (HR > 0)
- Notebook có ≥ 3 chart cho final_report

---

#### 🟥 T20 · Tab CF + cold-start fallback

| Field | Value |
|-------|-------|
| **Owner** | Tân Dư |
| **Day** | T4 30/07 (0.5 ngày) |
| **Depends on** | T18 |
| **Deliverable** | Tab CF trong `src/app.py` |

**Mô tả:**
Wire CF vào Streamlit tab thứ 3. Load artifact (không build mỗi click). Khi user không có trong train set hoặc không có candidate → fallback Simple với `st.info`.

**Acceptance:**
- userId=1 (existing) → top-K cá nhân hóa, không có phim đã rating
- userId=999999 (unknown) → message + fallback Simple top-K
- Click không freeze (load artifact, không rebuild)

---

#### 🟥 T21 · Regression full + final report + rehearsal

| Field | Value |
|-------|-------|
| **Owner** | Cả nhóm |
| **Day** | T4 30/07 (1 ngày) |
| **Depends on** | T19, T20 |
| **Deliverable** | `reports/final_report.md`, slide, demo 10 phút |

**Mô tả:**
Final regression test trên môi trường hoàn chỉnh, mọi case P0/P1 xanh. Mỗi role điền 1 section trong final_report. Rehearsal demo.

**Acceptance:**
- `pytest tests/ -v` exit 0, mọi case P0/P1 xanh
- Final report đủ 9 section (theo skeleton)
- Demo 10 phút ổn, không crash

---

## Ownership Matrix

```
              T5    T6    CN    T2    T3    T4
Tân Dư       watch watch T13         ─     T20,T21
Tâm          T10   ─     review     ─     ─     report
Duong        T11   T12   support   T16   ─     report
Loan         ─     ─     prep      T15   T17,T18 T19
Kiên         ─     ─     T14       prep  ─     T19,T21
```

---

## Dependencies Graph

```
T00 (done)
 │
 ├─→ T10 (Data clean)
 │    │
 │    ├─→ T11 (Simple)   ──┐
 │    ├─→ T12 (Content)  ──┤
 │    │                     ├─→ T13 (Demo v1) ─→ T14 (Test v1)
 │    │                     │
 │    ├─→ T15 (Utility)  ─→ T17 (ItemSim) ─→ T18 (Recommend) ─┐
 │    │                                                         │
 │    └─→ T16 (Overlap@K)                                      │
 │                                                              ▼
 │                                            T19 (Eval) ◄── T18
 │                                                │
 │                                                ▼
 │                                            T20 (Tab CF)
 │                                                │
 │                                                ▼
 └───────────────────────────────────────►  T21 (Ship) ◄── All
```

---

## Definition of Done (9 criteria, spec §14)

- [ ] Tài liệu mô tả rõ 3 phương pháp (final_report.md)
- [ ] Pipeline xử lý movies.csv và ratings.csv
- [ ] Simple top-K theo `weighted_rating` (filter `num_ratings >= m`)
- [ ] Content similar + genre explain
- [ ] CF theo `userId`, không gồm phim đã rating
- [ ] Cold-start → fallback Simple
- [ ] Streamlit 3 tab
- [ ] Test report + eval HR@10/NDCG@10 (hoặc sample)
- [ ] Không OOM (sparse + artifacts)

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| `ratings.csv` 647MB → OOM | dtype `int32/float32`, sparse matrix, sample khi dev |
| Pivot dense → crash | `scipy.sparse.csr_matrix`, không bao giờ dense |
| User mới không có rating | Fallback Simple Recommender |
| T10 trễ → block 6 task downstream | Schedule T10 ngày đầu (T5), một mình Tâm focus |
| T18 + T19 dồn sát | Buffer 0.5 ngày, prep từ CN |

---

## Tài liệu tham khảo

- Spec gốc: [`document.pdf`](../document.pdf)
- Board chính: [`TEAM_BOARD.md`](../TEAM_BOARD.md)
- Chi tiết role: [`plans/roles/`](../plans/roles/)
- Onboarding: [`docs/onboarding.md`](../docs/onboarding.md)
- Import Trello: [`plans/trello-cards.csv`](trello-cards.csv)

---

> **Liên hệ:** Tân Dư (Tech Lead) — duyệt code/model · Hoàng Đức Kiên (QA) — duyệt test/bug
>
> *Cập nhật: 22/07/2026 · Phiên bản: v1.0*
