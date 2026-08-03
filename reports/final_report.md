# Final Report — Movie Recommendation System

Owner: Tech Lead + cả nhóm · Due: T4 30/07/2026 23:59

> **Hướng dẫn điền:** mỗi section có tên owner phụ trách. Owner điền nội dung,
> giữ heading. Reference spec: `document.pdf` mục tương ứng.

## 1. Giới thiệu & mục tiêu

Owner: **Tân Dư**

- Bối cảnh project (PDF mục 1)
- 3 câu hỏi hệ thống trả lời (PDF mục 1: user mới / similar / personalized)
- Scope 3 phương pháp (PDF mục 3)
- Out of scope (Won't-have — PDF mục 10)

TODO Tân Dư: 1 đoạn giới thiệu + bullet 3 mục tiêu.

## 2. Dataset & preprocessing

Owner: **Trần Hoàng Minh Tâm**

- MovieLens 25M: movies.csv (62k phim, 2.9MB) + ratings.csv (25M ratings, 647MB)
- Schema (PDF mục 2.1, 2.2) — tham chiếu `docs/data-dictionary.md`
- Pipeline clean: tách year, genres_text, filter user≥20/movie≥50 (PDF mục 5)
- Bảng số liệu trước/sau filter — copy từ `reports/eda_summary.md`

| | Trước filter | Sau filter | % giữ lại |
|---|---|---|---|
| Số phim | 59.047 | 32.711 (content) - 13.172 (cf) | ~55.4% & ~22.3% |
| Số user | 162.541 | 162.516 (content) - 162.242 (cf) | ~99.98 & ~ 99.8%|
| Số rating | 25.000.095 | 24.945.390 (content) - 24.639.412 (cf) | ~99.78% & ~99.56% |
| Sparsity | ~99.74% | ~99.53% (content) - ~98.847 (cf) | — |
| Rating mean (C) | ~3.534 | ~3.535 (content) - ~3.54 (cf) | — |

TODO Tâm: điền bảng trên, copy số từ notebook 02.

## 3. Simple Recommender

Owner: **tran Duong**

- Mục tiêu (PDF mục 3.1): top-K phổ biến cho user mới
- Phương pháp: IMDb weighted rating
  - Công thức: `WR = (v/(v+m))*R + (m/(v+m))*C`
  - `m = quantile(0.80)`, `C = mean toàn dataset`
  - Filter `num_ratings >= m` trước khi rank
- Demo: tab Simple trong Streamlit
- Ưu điểm / hạn chế (PDF mục 3.1)
- Bảng top-10 phim (copy từ app)

TODO Duong: điền công thức + bảng top-10.

## 4. Content-based Recommender

Owner: **tran Duong**

- Mục tiêu (PDF mục 3.2): phim tương tự theo genre
- Phương pháp: CountVectorizer + cosine similarity (sparse, on-demand)
- Demo: tab Content với "Toy Story (1995)", "Heat (1995)"
- Genre overlap@K metric (PDF mục 6.2)

| Input | Genre overlap@10 |
|-------|------------------|
| Toy Story (1995) | TBD |
| Heat (1995) | TBD |
| Pulp Fiction (1994) | TBD |
| The Matrix (1999) | TBD |
| Inception (2010) | TBD |

TODO Duong: điền bảng trên, copy từ notebook 03.

## 5. Collaborative Filtering

Owner: **18- Thanh Loan**

- Mục tiêu (PDF mục 3.3): cá nhân hóa theo userId
- Phương pháp: Item-based CF, sparse csr_matrix, cosine item-item
- Pipeline (PDF mục 4):
  - load → clean → sparse utility → item similarity → recommend
- Cold-start fallback: user mới → Simple
- Demo: tab CF với userId=1, userId=42

TODO Loan: điền công thức + mô tả artifact (shape, density) từ `scripts/build_cf_artifacts.py`.

### 5.1 Evaluation

| Metric | Value | Sample size | Notes |
|--------|-------|-------------|-------|
| HR@10 | TBD | 200 | leave-last-out by timestamp |
| NDCG@10 | TBD | 200 | |

TODO Loan: copy từ `reports/cf_evaluation.md`.

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

- Streamlit Recommend Workbench: chọn/nhập user id, chọn top_k, Predict, bảng kết quả và context rating history
- Hạn chế (PDF mục 11 rủi ro):
  - Content chỉ có genre → gợi ý thô
  - CF cần user có rating → cold-start
  - Hybrid hiện tại kết hợp CF + Content bằng weighted score, chưa phải Neural CF

## 8. Test & Evaluation

Owner: **Hoàng Đức Kiên**

- Test strategy: unit (pytest) + manual smoke (Streamlit)
- Tổng quan: `reports/test_report.md`
- Eval: `reports/cf_evaluation.md` + chart từ `notebooks/05_evaluation_analysis.ipynb`

TODO Kiên: 1 đoạn tóm tắt + tham chiếu 2 report.

## 9. Phân công & retrospect

Owner: **Tân Dư** + cả nhóm

- Bảng ownership (copy từ TEAM_BOARD.md roster)
- Timeline thực tế so với plan
- Bài học cho sprint sau

## Appendix

- `document.pdf` — spec gốc
- `TEAM_BOARD.md` — Kanban
- `plans/roles/` — chi tiết task từng người
- `reports/eda_summary.md` — số liệu EDA
- `reports/test_report.md` — kết quả test
- `reports/cf_evaluation.md` — kết quả eval
