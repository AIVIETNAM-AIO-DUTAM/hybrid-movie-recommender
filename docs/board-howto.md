# How to update the board

Board chuẩn = **5 cột** (giống tool Kanban của nhóm):

1. **Chưa làm**
2. **Đang làm**
3. **Đang gặp vấn đề**
4. **Chờ duyệt**
5. **Hoàn thành**

## Flow hàng ngày

1. Sáng: mỗi người cập nhật cột task trên tool Kanban + sync 1 dòng vào `TEAM_BOARD.md` Snapshot nếu cần.
2. Bắt đầu làm → kéo **Chưa làm → Đang làm**.
3. Kẹt → kéo **Đang gặp vấn đề** + ghi lý do + tag người unblock.
4. Xong phần mình → kéo **Chờ duyệt** (không tự nhảy Hoàn thành).
5. Tân Dư / Hoàng Đức Kiên duyệt OK → kéo **Hoàn thành**.

## Ai duyệt gì

| Loại | Reviewer |
|------|----------|
| Code model / pipeline / data / demo | Tân Dư |
| Test case / bug / regression | Hoàng Đức Kiên |

## Sync MD ↔ tool

- Source of truth cho sprint checklist: `TEAM_BOARD.md`
- Source of truth vận hành kéo thẻ realtime: tool Kanban của nhóm
- Khi lệch: ưu tiên tool, rồi cập nhật lại MD cuối ngày
