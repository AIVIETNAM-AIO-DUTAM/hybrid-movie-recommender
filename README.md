# Movie Recommendation System (Module 2)

Hệ thống gợi ý phim trong **2 tuần** — 3 phương pháp:

1. **Simple Recommender** — top phim theo weighted rating (cold-start)
2. **Content-based** — phim tương tự theo genre
3. **Collaborative Filtering** — item-based CF theo `userId`

Tài liệu gốc: [`document.pdf`](./document.pdf)

---

## Bắt đầu nhanh (mỗi người)

```bash
cd "/Users/macbook/Module 2"
git lfs install                      # one-time per machine — fetch parquet fixtures qua LFS
git lfs pull                         # pull tests/fixtures/data/processed/*.parquet
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **LFS required:** `.parquet` fixtures (dùng bởi `tests/test_data_pipeline.py` D1–D3 và
> `tests/test_app_smoke.py` A1) được lưu qua Git LFS. Nếu clone mà chưa chạy `git lfs pull`,
> các file này chỉ là 130-byte pointer text → `pd.read_parquet()` sẽ crash với `ArrowInvalid`.
> Fallback nếu không có LFS: `python tests/fixtures/build_test_assets.py`.

1. Đọc [`TEAM_BOARD.md`](./TEAM_BOARD.md) — board chung, cập nhật status tại đây
2. Mở file role của mình trong [`plans/roles/`](./plans/roles/)
3. Làm đúng file mình **own** (xem bảng ownership bên dưới)
4. Đặt dataset vào `data/raw/` (`movies.csv`, `ratings.csv`)

### Chạy demo (sau khi có data + model)

```bash
streamlit run src/app.py
```

### Chạy test

```bash
pytest tests/ -v
```

---

## Ownership (tránh sửa chồng file)

| Role | Người | File sở hữu |
|------|--------|-------------|
| Tech Leader | **Tân Dư** | `src/app.py`, `TEAM_BOARD.md`, `docs/`, integration |
| AI Engineer (Data) | **Trần Hoàng Minh Tâm** | `src/data_processing.py`, `notebooks/01_*`, `notebooks/02_*`, `data/` |
| AI Engineer (Model) | **tran Duong** | `src/recommender_simple.py`, `src/recommender_content.py` |
| AI Engineer (Pipeline) | **18- Thanh Loan** | `src/recommender_cf.py`, `src/evaluation.py`, `artifacts/` CF |
| QA / Reviewer | **Hoàng Đức Kiên** | `tests/*`, `reports/test_report.md` |

Shared đọc được, nhưng **chỉ owner mới sửa**.

---

## Cấu trúc thư mục

```text
Module 2/
├── TEAM_BOARD.md          # Kanban — cập nhật mỗi ngày
├── document.pdf           # Spec gốc
├── requirements.txt
├── plans/
│   ├── plan.md
│   └── roles/             # Task chi tiết từng người
├── data/
│   ├── raw/               # movies.csv, ratings.csv (không commit)
│   └── processed/         # parquet sau clean
├── notebooks/
├── src/
├── artifacts/
├── tests/
├── reports/
└── docs/
```

---

## Quy ước cập nhật board (5 cột)

Đồng bộ với Kanban nhóm:

1. **Chưa làm** → 2. **Đang làm** → 3. **Đang gặp vấn đề** → 4. **Chờ duyệt** → 5. **Hoàn thành**

Chi tiết: [`TEAM_BOARD.md`](./TEAM_BOARD.md) + [`docs/board-howto.md`](./docs/board-howto.md).

Sync ngắn mỗi ngày: **15 phút**, mỗi người cập nhật cột task của mình.
