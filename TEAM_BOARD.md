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

## Snapshot hôm nay (D1 24/07)

| Người | Focus hôm nay | Cột | Blocker |
|-------|---------------|-----|---------|
| Tân Dư | **T01** setup repo + onboard 5 người | Chưa làm → Đang làm | — |
| Trần Hoàng Minh Tâm | **T01** phụ setup + load dataset check schema | Chưa làm → Đang làm | — |
| tran Duong | Đọc stubs Simple/Content, chờ T03 xong parquet | Chưa làm | Chờ T03 |
| 18- Thanh Loan | Đọc stubs CF, plan artifact | Chưa làm | — |
| Hoàng Đức Kiên | Đọc test stubs + test_cases.md | Chưa làm | — |

---

## Kanban (5 cột)

### 1. Chưa làm

| ID | Task | Assignee | Start | End | Depends | Phase |
|----|------|----------|-------|-----|---------|-------|
| T01 | Setup môi trường + onboard + dataset | Tân Dư, Tâm | D1 24/07 | D1 24/07 | — | 1 |
| T02 | EDA movies + ratings (notebooks 01–02) | Trần Hoàng Minh Tâm | D2 25/07 | D2 25/07 | T01 | 1 |
| T03 | Clean → parquet + data dictionary | Trần Hoàng Minh Tâm | D3 26/07 | D3 26/07 | T02 | 1 |
| T04 | Simple Recommender (weighted rating) | tran Duong | D4 27/07 | D5 28/07 | T03 | 1 |
| T05 | Content-based Recommender (genre cosine) | tran Duong | D4 27/07 | D5 28/07 | T03 | 1 |
| T06 | CF utility matrix | 18- Thanh Loan | D4 27/07 | D5 28/07 | T03 | 1 |
| T07 | CF item similarity | 18- Thanh Loan | D6 29/07 | D6 29/07 | T06 | 1 |
| T08 | CF recommend_for_user() + exclude seen | 18- Thanh Loan | D6 29/07 | D7 30/07 | T07 | 1 |
| T09 | App integration 3 tab + cold-start | Tân Dư | D7 30/07 | D8 31/07 | T04, T05, T08 | 1 |
| T10 | Eval HR@10 / NDCG@10 | 18- Thanh Loan, Hoàng Đức Kiên | D8 31/07 | D9 01/08 | T08 | 1 |
| T11 | Regression full + bug list | Hoàng Đức Kiên | D9 01/08 | D9 01/08 | T09 | 1 |
| T12 | Final report (9 section) + genre overlap@K | Cả nhóm | D10 02/08 | D11 03/08 | T11 | 2 |
| T13 | Charts/analysis notebook + slides | Loan, Kiên, Tân Dư | D11 03/08 | D12 04/08 | T12 | 2 |
| T14 | Rehearsal + ship | Cả nhóm | D13 05/08 | D14 06/08 | T13 | 2 |

### 2. Đang làm / 3. Đang gặp vấn đề / 4. Chờ duyệt

| ID | Task | Assignee | Note |
|----|------|----------|------|
| — | — | — | — |

### 5. Hoàn thành

| ID | Task | Assignee | Done | Evidence |
|----|------|----------|------|----------|
| **Pre-sprint** | Repo scaffold + role files + spec/plan review | Assistant | 22/07 | Stubs + 10 tests xanh |

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

- [ ] Tài liệu mô tả rõ 3 phương pháp (final_report.md)
- [ ] Pipeline xử lý movies.csv và ratings.csv (data_processing + run_pipeline.py)
- [ ] Simple top-K theo `weighted_rating` (lọc `num_ratings >= m`)
- [ ] Content similar + genre explain
- [ ] CF theo `userId`, không gồm phim đã rating
- [ ] Cold-start → fallback Simple
- [ ] Streamlit 3 tab
- [ ] Test report + eval HR@10/NDCG@10 (hoặc sample)
- [ ] Không OOM (sparse + artifacts)

---

## Links

- Spec: [document.pdf](./document.pdf)
- Trello: [plans/trello-import.md](./plans/trello-import.md) · [plans/trello-cards.csv](./plans/trello-cards.csv)
- Roles: [plans/roles/](./plans/roles/)
- Onboarding: [docs/onboarding.md](./docs/onboarding.md)
- Sprint report (PDF): [reports/sprint_plan.html](./reports/sprint_plan.html)
