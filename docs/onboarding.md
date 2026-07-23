# Onboarding — vào làm trong 15 phút

## 1. Clone / mở repo

Mở folder `Module 2` trong Cursor/VS Code.

## 2. Setup môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Nhận role

Tech Lead gán tên vào `TEAM_BOARD.md` và file trong `plans/roles/`.

| File | Role |
|------|------|
| `plans/roles/01-tech-lead.md` | Tech Lead |
| `plans/roles/02-data-engineer.md` | Data Engineer |
| `plans/roles/03-ml-simple-content.md` | ML A |
| `plans/roles/04-ml-collaborative-filtering.md` | ML B |
| `plans/roles/05-qa-tester.md` | QA |

## 4. Dataset

Đặt file Kaggle vào:

```text
data/raw/movies.csv
data/raw/ratings.csv
```

> File CSV lớn **không** commit lên git (đã có trong `.gitignore`).

## 5. Quy tắc làm việc

1. Chỉ sửa file mình own.
2. Trước khi code: chuyển task sang `doing` trên `TEAM_BOARD.md`.
3. Xong: chuyển `done` + ghi evidence (commit hash / screenshot / path artifact).
4. Blocker > 2 giờ → ping Tech Lead + ghi vào cột Blocked.
5. Không đẩy secrets / full `ratings.csv` lên remote nếu repo public.

## 6. Check nhanh môi trường

```bash
python -c "import pandas, sklearn, scipy, streamlit; print('OK')"
pytest tests/ -q
```

Stub test có thể fail một phần cho đến khi model được implement — QA sẽ cập nhật kỳ vọng.
