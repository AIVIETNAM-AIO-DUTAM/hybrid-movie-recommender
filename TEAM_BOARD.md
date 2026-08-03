# TEAM BOARD — Movie Recommendation System

> Đồng bộ Trello 5 cột: **Chưa làm → Đang làm → Đang gặp vấn đề → Chờ duyệt → Hoàn thành**
> Import cards: [plans/trello-import.md](./plans/trello-import.md) · [plans/trello-cards.csv](./plans/trello-cards.csv)

**Sprint:** 24/07/2026 → 06/08/2026 (14 ngày, 2 tuần) · **Team:** 5 người
**🔒 APP FREEZE:** 01/08/2026 — sau đó chỉ làm report

---

## Team roster

| Người | Role | File role |
|-------|------|-----------|
| **Tân Dư** | Tech Leader | [01-tech-lead.md](./plans/roles/01-tech-lead.md) |
| **Trần Hoàng Minh Tâm** | AI Engineer (Data) | [02-data-engineer.md](./plans/roles/02-data-engineer.md) |
| **tran Duong** | AI Engineer (Model) | [03-ml-simple-content.md](./plans/roles/03-ml-simple-content.md) |
| **18- Thanh Loan** | AI Engineer (Pipeline) | [04-ml-collaborative-filtering.md](./plans/roles/04-ml-collaborative-filtering.md) |
| **Hoàng Đức Kiên** | QA / Reviewer | [05-qa-tester.md](./plans/roles/05-qa-tester.md) |

---

## Timeline (14 ngày)

| Phase | Từ | Đến | Mục tiêu |
|-------|----|----|----------|
| **1 — App Build** | T5 24/07 | T6 **01/08** | Parquet + Simple + Content + CF + Demo 3 tab + Eval + **🔒 APP FREEZE** |
| **2 — Report & Ship** | T7 02/08 | T4 06/08 | Final report (9 section) + charts + slides + rehearsal + ship |

**Meeting:** T5 24/07 (kickoff) · CN 27/07 (review build) · CN 03/08 (review report) · T4 06/08 (ship)

---

## Snapshot hôm nay (D8 31/07)

| Người | Focus hôm nay | Cột | Blocker |
|-------|---------------|-----|---------|
| Tân Dư | T09 app freeze prep · fix load_processed unpack | Đang làm | — |
| Trần Hoàng Minh Tâm | Pipeline parquet local OK | Hoàn thành | — |
| tran Duong | genre_overlap_at_k + overlap bảng report | Hoàn thành | — |
| 18- Thanh Loan | CF artifacts + eval numbers | Hoàn thành | — |
| Hoàng Đức Kiên | T11 regression unskip D/A + test_report | Đang làm | — |

Branch gap-fix: `fix/sprint-gaps-d8`

---

## Kanban (5 cột)

### 1. Chưa làm

| ID | Task | Assignee | Start | End | Depends | Phase |
|----|------|----------|-------|-----|---------|-------|
| T12 | Final report (9 section) + genre overlap@K | Cả nhóm | D10 02/08 | D11 03/08 | T11 | 2 |
| T13 | Charts/analysis notebook + slides | Loan, Kiên, Tân Dư | D11 03/08 | D12 04/08 | T12 | 2 |
| T14 | Rehearsal + ship | Cả nhóm | D13 05/08 | D14 06/08 | T13 | 2 |

### 2. Đang làm / 3. Đang gặp vấn đề / 4. Chờ duyệt

| ID | Task | Assignee | Note |
|----|------|----------|------|
| T11 | Regression full + bug list | Kiên | Branch `fix/sprint-gaps-d8` — chờ duyệt trước APP FREEZE |

### 5. Hoàn thành

| ID | Task | Assignee | Done | Evidence |
|----|------|----------|------|----------|
| **Pre-sprint** | Repo scaffold + role files + spec/plan review | Assistant | 22/07 | Stubs + 10 tests xanh |
| T01 | Setup môi trường + onboard + dataset | Tân Dư, Tâm | 24/07 | README + raw data |
| T02 | EDA movies + ratings | Tâm | 25/07 | notebooks 01–02, eda_summary |
| T03 | Clean → parquet + data dictionary | Tâm | 26/07 | ratings_cf / ratings_content parquet |
| T04 | Simple Recommender | Duong | 28/07 | recommender_simple.py |
| T05 | Content-based + genre_overlap_at_k | Duong | 31/07 | recommender_content.py |
| T06–T08 | CF utility / similarity / recommend | Loan | 31/07 | artifacts/*.npz |
| T09 | App 3 tab + cold-start | Tân Dư | 31/07 | src/app.py |
| T10 | Eval HR@10 / NDCG@10 | Loan, Kiên | 31/07 | cf_evaluation.md + evaluation/ |

---

## Quy tắc kéo thẻ

| Khi… | Cột |
|------|-----|
| Chưa bắt đầu | Chưa làm |
| Đang làm | Đang làm |
| Kẹt > 2 giờ | Đang gặp vấn đề |
| Xong, chờ review | Chờ duyệt |
| Duyệt OK | Hoàn thành |

Reviewer: code/demo → **Tân Dư** · test/bug → **Hoàng Đức Kiên**

---

## Definition of Done (khớp `document.pdf` mục 14)

- [ ] Tài liệu mô tả rõ 3 phương pháp (final_report.md) — skeleton + số liệu đã điền một phần; prose Phase 2
- [x] Pipeline xử lý movies.csv và ratings.csv (data_processing + run_pipeline.py)
- [x] Simple top-K theo `weighted_rating` (lọc `num_ratings >= m`)
- [x] Content similar + genre explain (+ `genre_overlap_at_k`)
- [x] CF theo `userId`, không gồm phim đã rating
- [x] Cold-start → fallback Simple
- [x] Streamlit 3 tab
- [x] Test report + eval HR@10/NDCG@10 (hoặc sample)
- [x] Không OOM (sparse + top-K artifacts)

---

## Links

- Spec: [document.pdf](./document.pdf)
- Trello: [plans/trello-import.md](./plans/trello-import.md) · [plans/trello-cards.csv](./plans/trello-cards.csv)
- Roles: [plans/roles/](./plans/roles/)
- Onboarding: [docs/onboarding.md](./docs/onboarding.md)
- Sprint report (PDF): [reports/sprint_plan.html](./reports/sprint_plan.html)
