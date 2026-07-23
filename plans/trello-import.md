# Trello Import — Movie Recommendation System

> Sprint: **24/07/2026 → 06/08/2026** (14 ngày, 2 tuần)
> Board columns: Chưa làm → Đang làm → Đang gặp vấn đề → Chờ duyệt → Hoàn thành
> **🔒 APP FREEZE: 01/08/2026** — sau đó chỉ làm report
> **Meeting:** T5 24/07 (kickoff) · CN 27/07 (review build) · CN 03/08 (review report)

## Thành viên

| Tên trên Trello | Role |
|-----------------|------|
| Tân Dư | Tech Leader |
| Trần Hoàng Minh Tâm | AI Engineer (Data) |
| tran Duong | AI Engineer (Model) |
| 18- Thanh Loan | AI Engineer (Pipeline) |
| Hoàng Đức Kiên | QA / Reviewer |

---

## Phase 1 — App Build (D1–D9, T5 24/07 → T6 01/08)

| Card title | List | Members | Start | Due | Est | Labels | Description |
|------------|------|---------|-------|-----|-----|--------|-------------|
| `[T01] Setup môi trường + onboard` | Chưa làm | Tân Dư;Trần Hoàng Minh Tâm | 2026-07-24 | 2026-07-24 | 1d | setup;lead;data | Repo + venv + dataset check + role files + board |
| `[T02] EDA movies + ratings` | Chưa làm | Trần Hoàng Minh Tâm | 2026-07-25 | 2026-07-25 | 1d | data;eda | Notebooks 01, 02: schema, sparsity, genre dist, duplicate |
| `[T03] Clean → parquet + data dictionary` | Chưa làm | Trần Hoàng Minh Tâm | 2026-07-26 | 2026-07-26 | 1d | data | parquet + filter user≥20 movie≥50 + data-dictionary.md |
| `[T04] Simple Recommender (weighted rating)` | Chưa làm | tran Duong | 2026-07-27 | 2026-07-28 | 1.5d | model | IMDb WR; build_movie_stats + recommend_top_movies |
| `[T05] Content-based (genre cosine)` | Chưa làm | tran Duong | 2026-07-27 | 2026-07-28 | 1.5d | model | CountVectorizer sparse; recommend_similar_movies |
| `[T06] CF utility matrix` | Chưa làm | 18- Thanh Loan | 2026-07-27 | 2026-07-28 | 1.5d | pipeline;cf | csr_matrix user×movie; save utility.npz + maps.pkl |
| `[T07] CF item-item similarity` | Chưa làm | 18- Thanh Loan | 2026-07-29 | 2026-07-29 | 1d | pipeline;cf | cosine_similarity sparse; save item_similarity.npz |
| `[T08] CF recommend_for_user()` | Chưa làm | 18- Thanh Loan | 2026-07-29 | 2026-07-30 | 1.5d | pipeline;cf | Top-K userId; sparse matmul; exclude seen; raise KeyError/ValueError |
| `[T09] App integration 3 tab + cold-start` | Chưa làm | Tân Dư | 2026-07-30 | 2026-07-31 | 1.5d | lead;demo | Wire 3 tab app.py; fallback Simple cho user mới |
| `[T10] Eval HR@10 / NDCG@10` | Chưa làm | 18- Thanh Loan;Hoàng Đức Kiên | 2026-07-31 | 2026-08-01 | 1.5d | qa;cf | Leave-last-out sample 200; report + charts |
| `[T11] Regression full + bug list` | Chưa làm | Hoàng Đức Kiên | 2026-08-01 | 2026-08-01 | 1d | qa | **🔒 APP FREEZE** · Full regression P0/P1 xanh |

## Phase 2 — Report & Ship (D10–D14, T7 02/08 → T4 06/08)

| Card title | List | Members | Start | Due | Est | Labels | Description |
|------------|------|---------|-------|-----|-----|--------|-------------|
| `[T12] Final report (9 section) + overlap@K` | Chưa làm | Tân Dư;Trần Hoàng Minh Tâm;tran Duong;18- Thanh Loan;Hoàng Đức Kiên | 2026-08-02 | 2026-08-03 | 1.5d | docs | Mỗi role 1 section + overlap@K metric |
| `[T13] Charts/analysis notebook + slides` | Chưa làm | 18- Thanh Loan;Hoàng Đức Kiên;Tân Dư | 2026-08-03 | 2026-08-04 | 1.5d | docs;demo | Notebook 05 + slide outline + screenshots |
| `[T14] Rehearsal + ship` | Chưa làm | Tân Dư;Trần Hoàng Minh Tâm;tran Duong;18- Thanh Loan;Hoàng Đức Kiên | 2026-08-05 | 2026-08-06 | 1.5d | demo | Demo 10 phút + polish + nộp |

CSV: [`trello-cards.csv`](./trello-cards.csv)

---

## Timeline theo ngày

| Ngày | Việc chính | Người |
|------|-----------|-------|
| **D1 T5 24/07** | **HỌP kickoff** + T01 setup | Tân Dư, Tâm |
| D2 T6 25/07 | T02 EDA | Tâm |
| D3 T7 26/07 | T03 Clean parquet (bottleneck) | Tâm |
| **D4 CN 27/07** | **HỌP review** + T04/T05/T06 parallel | Duong, Loan, Tâm |
| D5 T2 28/07 | T04/T05/T06 finish | Duong, Loan |
| D6 T3 29/07 | T07 similarity + T08 recommend | Loan |
| D7 T4 30/07 | T08 finish + T09 App integration | Tân Dư |
| D8 T5 31/07 | T09 finish (APP COMPLETE) + T10 Eval | Tân Dư, Loan, Kiên |
| **D9 T6 01/08** | T10 finish + T11 Regression + **🔒 APP FREEZE** | Loan, Kiên |
| D10 T7 02/08 | T12 Final report draft | Cả nhóm |
| **D11 CN 03/08** | **HỌP review** + T12 finish + T13 Charts | Cả nhóm |
| D12 T2 04/08 | T13 finish + Slides | Loan, Kiên, Tân Dư |
| D13 T3 05/08 | Polish + cross-review | Cả nhóm |
| **D14 T4 06/08** | **T14 Ship** | Cả nhóm |

---

## Dataset đã tải

Nguồn: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/)

```text
data/raw/movies.csv   ~2.9 MB
data/raw/ratings.csv  ~647 MB
```

Schema đúng PDF: `movieId,title,genres` · `userId,movieId,rating,timestamp`
