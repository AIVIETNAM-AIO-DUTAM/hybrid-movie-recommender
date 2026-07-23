# Role 01 — Tech Lead / App Integration

**Owner:** Tân Dư
**Owns:** `src/app.py`, `TEAM_BOARD.md`, `docs/`, integration, demo script, final slide outline

## Trách nhiệm

- Chốt scope / DoD / ownership
- Ghép 3 recommender vào Streamlit (`src/app.py`) — **deadline 01/08 (D8)**
- Cold-start fallback: user mới → Simple
- Phase 2: tổng hợp report + demo cuối

## Phase 1 — App Build (D1–D9)

### D1 T5 24/07 — **HỌP KICKOFF** + T01 Setup
- [ ] Có mặt họp kickoff, confirm ownership 5 người
- [ ] Verify repo scaffold + role files
- [ ] Verify dataset MovieLens 25M trong `data/raw/`

### D7 T4 30/07 — T09 App integration 3 tab (bắt đầu)

**Mục tiêu:** Wire 3 tab Streamlit, smoke test trên máy demo.

**Sub-task checklist:**
- [ ] Tab Simple: slider top_k + filter genre → `recommend_top_movies()`
- [ ] Tab Content: cached `build_content_model()` + `recommend_similar_movies()`
- [ ] Tab CF: cached `load_cf_cached()` + `recommend_for_user()` (cần T08 xong)
- [ ] Try/except đa nhánh (KeyError + ValueError + Exception) → fallback Simple
- [ ] Empty result → `st.info` + fallback Simple
- [ ] Smoke test userId=1, userId=999999, "Toy Story (1995)"
- [ ] Screenshot cho evidence

**File cần làm:**
- `src/app.py` — wire 3 tab đầy đủ

### D8 T5 31/07 — T09 finish (APP COMPLETE)

**Sub-task checklist:**
- [ ] 3 tab chạy ổn định trên máy demo
- [ ] Cold-start fallback verify 2 case (user mới + user không có rating ≥ 4.0)
- [ ] Hand-off cho Kiên regression D9
- [ ] ⚠️ Từ 01/08 không thêm feature, chỉ sửa critical bug

**Done khi (T09):** App 3 tab chạy ổn, fallback working. Evidence: 5 screenshot (1 mỗi tab + 2 fallback).

### D9 T6 01/08 — **🔒 APP FREEZE**

- [ ] Có mặt hỗ trợ Kiên khi regression phát hiện critical bug
- [ ] Sau 01/08 → chuyển sang Phase 2

## Phase 2 — Report & Ship (D10–D14)

### D10 T7 02/08 — T12 Final report draft (start)

**Sub-task checklist:**
- [ ] Mở `reports/final_report.md` (skeleton 9 section đã có)
- [ ] Section 1 (Intro) + 6 (Limitations) + 7 (Future) + 9 (Conclusion): Tân Dư tự viết
- [ ] Phân công các section khác cho 4 role

**Files:**
- `reports/final_report.md` — section 1, 6, 7, 9

### D11 CN 03/08 — **HỌP REVIEW REPORT**

- [ ] Trình bày progress report
- [ ] Verify mọi section draft xong

### D11–D12 CN–T2 03–04/08 — T13 Slides

**Sub-task checklist:**
- [ ] Slide outline (10 phút demo, mỗi role 1 phút)
- [ ] Screenshot 3 tab app cho slide
- [ ] 5–7 slide tổng

**Files:**
- `reports/slides_outline.md` — outline + script
- `reports/screenshots/` — 5 screenshot

### D13 T3 05/08 — Polish + cross-review

- [ ] Cross-review report với cả nhóm
- [ ] Sửa typo / số liệu sai

### D14 T4 06/08 — T14 Rehearsal + ship
- [ ] Rehearsal 10 phút
- [ ] Nộp bản cuối

## Không làm

- Không sửa logic bên trong `recommender_*.py` trừ khi unblock khẩn
- Không rewrite data pipeline của Tâm
- Sau 01/08: **không thêm feature**

## Done khi (toàn sprint)

App 3 tab chạy ổn trước 01/08 23:59. Final report + slide rehearsal xong trước T4 06/08 23:59.
