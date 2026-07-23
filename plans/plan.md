---
title: Movie Recommendation System — 14 ngày (2 tuần)
status: in-progress
priority: P0
effort: high
branch: main
tags: [recommender, simple, content-based, collaborative-filtering]
created: 2026-07-22
updated: 2026-07-22
---

# Plan — Movie Recommendation System (14 ngày)

## Mục tiêu

Xây demo gợi ý phim với 3 phương pháp trong scope `document.pdf`:

1. Simple Recommender
2. Content-based Recommender
3. Item-based Collaborative Filtering

## Constraint quan trọng (mới)

> **App phải xong vào ngày 01/08 (D9).** Sau đó 5 ngày (02/08 → 06/08) chỉ tập trung report + polish.
> App freeze cuối ngày 01/08 — không thêm feature, chỉ sửa critical bug.

## Out of scope (Won't-have)

- Hybrid / Context-aware / Matrix Factorization / Neural CF
- A/B testing / production deploy

## Deviations from spec

- **CF model code + evaluation** own bởi role "AI Engineer (Pipeline)" (Loan) thay vì "AI Model" (PDF mục 8). Lý do: CF nặng nhất, Pipeline rảnh sau data prep.
- **`pipeline.py`** (PDF mục 12) gộp vào `recommender_cf.py`.
- **Pre-sprint setup** đã hoàn thành trước T01.

## Timeline 14 ngày (T5 24/07 → T4 06/08)

> **2 giai đoạn:**
> - **Phase 1 — App Build:** D1–D9 (24/07 → 01/08), 9 ngày
> - **Phase 2 — Report & Ship:** D10–D14 (02/08 → 06/08), 5 ngày
>
> **Meeting:** T5 24/07 (kickoff) · CN 27/07 (review mid-build) · CN 03/08 (review report) · T4 06/08 (ship)

| Day | Ngày | Tasks chính | Phase |
|-----|------|-------------|-------|
| D1  | T5 24/07 | **Meeting kickoff** · T01 Setup môi trường | 1 |
| D2  | T6 25/07 | T02 EDA movies + ratings | 1 |
| D3  | T7 26/07 | T03 Clean → parquet (bottleneck) | 1 |
| D4  | CN 27/07 | **Meeting review** · T04 Simple · T05 Content · T06 Utility matrix | 1 |
| D5  | T2 28/07 | T04/T05 finish · T06 finish | 1 |
| D6  | T3 29/07 | T07 Item similarity · T08 recommend · Demo v1 | 1 |
| D7  | T4 30/07 | T08 finish · T09 App integration 3 tab | 1 |
| D8  | T5 31/07 | T09 finish (APP COMPLETE) · T10 Eval start | 1 |
| **D9** | **T6 01/08** | T10 Eval finish · T11 Regression · **🔒 APP FREEZE** | 1 |
| D10 | T7 02/08 | T12 Final report draft (9 section) | 2 |
| D11 | CN 03/08 | **Meeting review** · T12 finish · T13 Charts/analysis | 2 |
| D12 | T2 04/08 | T13 finish · Slides | 2 |
| D13 | T3 05/08 | Polish + cross-review | 2 |
| D14 | T4 06/08 | **T14 Rehearsal + ship** | 2 |

## Task ownership summary

### Phase 1 — App Build (T01–T11)

| ID | Task | Owner | Day | Goal |
|----|------|-------|-----|------|
| T01 | Setup môi trường + onboard | Tân Dư + Tâm | D1 | repo + venv + dataset |
| T02 | EDA movies + ratings | Tâm | D2 | notebooks 01, 02 |
| T03 | Clean → parquet + data dictionary | Tâm | D3 | unblock 6 task |
| T04 | Simple Recommender (weighted rating) | Duong | D4–D5 | tab 1 OK |
| T05 | Content-based (genre cosine) | Duong | D4–D5 | tab 2 OK |
| T06 | CF utility matrix | Loan | D4–D5 | artifact sparse |
| T07 | CF item similarity | Loan | D6 | artifact |
| T08 | CF recommend_for_user | Loan | D6–D7 | tab 3 logic |
| T09 | App integration 3 tab + cold-start | Tân Dư | D7–D8 | app complete |
| T10 | Eval HR@10 / NDCG@10 | Loan + Kiên | D8–D9 | metric |
| T11 | Regression full + bug list | Kiên | D9 | **🔒 APP FREEZE** |

### Phase 2 — Report & Ship (T12–T14)

| ID | Task | Owner | Day | Goal |
|----|------|-------|-----|------|
| T12 | Final report (9 section) + Genre overlap@K | Cả nhóm | D10–D11 | report.md |
| T13 | Charts/analysis notebook + slides | Loan, Kiên, Tân Dư | D11–D12 | visuals |
| T14 | Rehearsal + ship | Cả nhóm | D13–D14 | demo + nộp |

## Critical path

```text
T03 (parquet) ─┬─→ T04 Simple ─────────────┐
               ├─→ T05 Content ────────────┤
               └─→ T06 Utility → T07 Sim → T08 Recommend → T09 App ─→ T11 FREEZE
```

T03 là bottleneck (T04/T05/T06 phụ thuộc). T06 phải bắt đầu D4 (sớm hơn so với plan cũ) để CF kịp app deadline 01/08.

## Pre-created stubs (signature + docstring + TODO)

### Code
| File | Owner |
|------|-------|
| `src/data_processing.py` | Tâm |
| `src/recommender_simple.py` | Duong |
| `src/recommender_content.py` | Duong |
| `src/recommender_cf.py` | Loan |
| `src/evaluation.py` | Loan |
| `src/app.py` | Tân Dư |
| `scripts/run_pipeline.py` | Tâm |
| `scripts/build_cf_artifacts.py` | Loan |
| `scripts/run_evaluation.py` | Loan |

### Notebooks
| File | Owner |
|------|-------|
| `notebooks/01_eda_movies.ipynb` | Tâm |
| `notebooks/02_eda_ratings.ipynb` | Tâm |
| `notebooks/03_modeling.ipynb` | Duong (overlap@K) |
| `notebooks/04_cf_experiments.ipynb` | Loan |
| `notebooks/05_evaluation_analysis.ipynb` | Kiên |

### Tests
| File | Cases |
|------|-------|
| `tests/test_recommender.py` | S1, S2, S3, C3, C4, F1, F2, F3, F4 |
| `tests/test_data_pipeline.py` | D1, D2, D3 |
| `tests/test_app_smoke.py` | A1, A2 |

## Role files

- [01-tech-lead.md](./roles/01-tech-lead.md)
- [02-data-engineer.md](./roles/02-data-engineer.md)
- [03-ml-simple-content.md](./roles/03-ml-simple-content.md)
- [04-ml-collaborative-filtering.md](./roles/04-ml-collaborative-filtering.md)
- [05-qa-tester.md](./roles/05-qa-tester.md)

## Acceptance criteria (DoD)

Khớp với `document.pdf` mục 14 (9 criteria). Xem đầy đủ tại `TEAM_BOARD.md`.
