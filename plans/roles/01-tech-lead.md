# Role 01 — Tech Lead / App Integration

**Owner:** Tân Dư
**Owns:** `src/app/streamlit_app.py`, `src/app/model_adapter.py`, `TEAM_BOARD.md`, `docs/`, integration, demo script, final slide outline

## Trách nhiệm

- Chốt scope / DoD / ownership
- Ghép Hybrid Recommender vào Streamlit (`src/app/streamlit_app.py`) — **deadline 01/08 (D8)**
- Tách model adapter để sau này thay model chính không phải sửa UI
- Phase 2: tổng hợp report + demo cuối

## Phase 1 — App Build (D1–D9)

### D1 T5 24/07 — **HỌP KICKOFF** + T01 Setup
- [ ] Có mặt họp kickoff, confirm ownership 5 người
- [ ] Verify repo scaffold + role files
- [ ] Verify dataset MovieLens 25M trong `data/raw/`

### D7 T4 30/07 — T09 App integration Hybrid (bắt đầu)

**Mục tiêu:** Wire giao diện Recommend Streamlit, smoke test trên máy demo.

**Sub-task checklist:**
- [x] Layout gồm chọn/nhập user id, chọn top_k, nút Predict, bảng kết quả, context rating history
- [x] Tách `model_adapter.py` để gọi Hybrid = CF + Content artifacts
- [x] App hỗ trợ fixture qua `REC_DATA_DIR` và `REC_MODEL_DIR`
- [ ] Smoke test userId thật, userId không có history, top_k khác nhau
- [ ] Screenshot cho evidence

**File cần làm:**
- `src/app/streamlit_app.py` — giao diện Streamlit
- `src/app/model_adapter.py` — adapter gọi model

### D8 T5 31/07 — T09 finish (APP COMPLETE)

**Sub-task checklist:**
- [ ] App Recommend chạy ổn định trên máy demo
- [ ] Verify user có history và user ít/không có tín hiệu
- [ ] Hand-off cho Kiên regression D9
- [ ] ⚠️ Từ 01/08 không thêm feature, chỉ sửa critical bug

**Done khi (T09):** App Recommend chạy ổn, model adapter trả đúng schema. Evidence: screenshot kết quả + context.

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
- [ ] Screenshot app Recommend cho slide
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

App Recommend chạy ổn trước 01/08 23:59. Final report + slide rehearsal xong trước T4 06/08 23:59.
