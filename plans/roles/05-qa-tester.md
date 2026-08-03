# Role 05 — QA / Tester

**Owner:** Hoàng Đức Kiên
**Owns:** `tests/test_cases.md`, `tests/test_recommender.py`, `tests/test_data_pipeline.py`, `tests/test_app_smoke.py`, `reports/test_report.md`, `notebooks/05_evaluation_analysis.ipynb`, bug log

## Trách nhiệm

- Viết test case từ acceptance criteria
- Test UI/API + edge cases
- Log bug, verify fix
- Regression full **D9 01/08 (APP FREEZE)**
- Phân tích số liệu evaluation, vẽ chart cho final report

## Phase 1 — App Build (D1–D9)

### D1 T5 24/07 — Prep
- [ ] Có mặt họp kickoff
- [ ] Đọc `tests/test_recommender.py` (10 tests đã xanh)
- [ ] Đọc `tests/test_cases.md`, `test_data_pipeline.py`, `test_app_smoke.py`

### D5 T2 28/07 — Test T04/T05 (unblock early)

**Mục tiêu:** Khi Simple/Content xong, chạy test ngay để tránh dồn cuối.

**Sub-task checklist:**
- [x] D1/D2/D3 trong `test_data_pipeline.py` đã chạy bằng fixture parquet
- [x] Chạy `pytest tests/ -v`
- [x] Verify S3 (columns schema) xanh
- [x] Cập nhật `tests/test_cases.md` status

### D7–D8 T4–T5 30–31/07 — Test T08/T09 (CF + app)

**Sub-task checklist:**
- [x] A1/A2 trong `test_app_smoke.py` đã chạy bằng fixture data/model
- [x] Verify 4 tests CF (F1–F4) xanh
- [ ] Test UI smoke app Hybrid trên máy demo
- [ ] Log bug vào `reports/test_report.md` nếu có

### D9 T6 01/08 — **T11 Regression + APP FREEZE** 🔒

**Mục tiêu:** Final regression trên môi trường hoàn chỉnh. Mọi case P0/P1 xanh. Sau đó app **freeze**.

**Sub-task checklist:**
- [x] Verify fixture artifacts (parquet + CF/Content artifacts) exist
- [x] Chạy `pytest tests/ -v`
- [x] Verify 10 tests recommender cũ vẫn xanh
- [x] Verify S3 mới xanh
- [x] Verify D1/D2/D3 + A1/A2 xanh, không còn skip
- [ ] Log mọi fail vào bug list
- [x] Cập nhật `reports/test_report.md` summary
- [ ] **🔒 APP FREEZE** — sau khi report xanh, không thêm feature

**Cases bắt buộc (PDF §3.1, §3.2, §3.3):**

| ID | Case | Expected |
|----|------|----------|
| S1 | Simple top_k=10 | đúng 10 rows |
| S2 | Phim ít rating điểm cao | không chiếm top |
| S3 | Simple columns schema | đủ 6 cột |
| C1 | Toy Story | nhiều Animation/Children |
| C2 | Heat | Action/Crime/Thriller |
| C3 | title không tồn tại | ValueError |
| C4 | không trả chính input | True |
| F1 | userId tồn tại | không gồm phim đã rating |
| F2 | userId không tồn tại | KeyError/fallback |
| F3 | sparse pipeline | không OOM |
| F4 | user không có liked | ValueError/fallback |
| D1 | movies_clean schema | đủ cột xử lý |
| D2 | year extracted | year hợp lệ |
| D3 | ratings fixture + dtype | dtype/range/no dup hợp lệ |
| A1 | model adapter predict | trả top-K đúng schema |
| A2 | Streamlit app import | import được layout hiện tại |

**Done khi (T11):** Test report xanh P0/P1. App freeze.

## Phase 2 — Report & Ship (D10–D14)

### D10 T7 02/08 — T12 Final report (section Test & Eval)

**Sub-task checklist:**
- [ ] Section 8 (Test & Evaluation) trong `reports/final_report.md`
- [ ] Điền: test count, pass rate, bug list summary, eval metric summary

**File:**
- `reports/final_report.md` — section 8

### D11–D12 CN–T2 03–04/08 — T13 Charts + slides

**Sub-task checklist:**
- [ ] `notebooks/05_evaluation_analysis.ipynb` — load `cf_eval_scores.csv`
- [ ] 3 chart: HR@10 theo top_k, NDCG@10 theo top_k, rating distribution
- [ ] Slide cho test report (1 slide)

**Files:**
- `notebooks/05_evaluation_analysis.ipynb` — 3 chart
- `reports/slides_outline.md` — slide test report

### D13–D14 — Polish + rehearsal
- [ ] Cross-review report
- [ ] Có mặt rehearsal D14

## Bug severity

- **P0**: crash demo / sai data leak (trả phim đã xem)
- **P1**: sai logic core
- **P2**: UX / copy / performance nhẹ

## Done khi (toàn sprint)

Test report xanh P0/P1 trước **01/08 (APP FREEZE)**. Trạng thái tự động
gần nhất: `15 passed`, `0 failed`, `0 skipped`. Eval chart + report section 8
xong D12. Rehearsal D14.
