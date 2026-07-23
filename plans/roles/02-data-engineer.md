# Role 02 — Data Engineer (AI Data)

**Owner:** Trần Hoàng Minh Tâm
**Owns:** `src/data_processing.py`, `scripts/run_pipeline.py`, `notebooks/01_eda_movies.ipynb`, `notebooks/02_eda_ratings.ipynb`, `data/`, `docs/data-dictionary.md`, `reports/eda_summary.md`

## Trách nhiệm

- Tải & xác nhận schema dataset
- EDA movies + ratings
- Clean → parquet, tối ưu dtype, lọc sparse user/movie
- Data dictionary
- **Deadline app:** T03 phải xong D3 26/07 (bottleneck cho 6 task)

## Phase 1 — App Build (D1–D9)

### D1 T5 24/07 — T01 Setup (phối Tân Dư)
- [ ] Có mặt họp kickoff
- [ ] Verify `data/raw/movies.csv` + `ratings.csv` đọc được bằng pandas
- [ ] Confirm schema đúng PDF mục 2.1, 2.2

### D2 T6 25/07 — T02 EDA movies + ratings

**Sub-task checklist:**
- [ ] Notebook 01: số phim, genre distribution, duplicate titles, year range
- [ ] Notebook 02: số user, số movie, sparsity %, rating histogram, top user/movie counts
- [ ] Ghi nhận số liệu vào `reports/eda_summary.md` (TBD → số thật)

**File cần làm:**
- `notebooks/01_eda_movies.ipynb` — điền cells
- `notebooks/02_eda_ratings.ipynb` — điền cells

### D3 T7 26/07 — T03 Clean → parquet + data dictionary **(BOTTLENECK)**

**Mục tiêu:** Xuất parquet để unblock 6 task (T04, T05, T06, …). **Phải xong cuối D3.**

**Sub-task checklist:**
- [ ] Implement `load_movies_raw(path)` — dtype tối ưu + friendly FileNotFoundError
- [ ] Implement `load_ratings_raw(path)` — usecols chỉ 4 cột cần
- [ ] Implement `clean_movies(movies)` — extract year, split genres, normalize sentinel
- [ ] Implement `clean_ratings(ratings, min_user=20, min_movie=50)` — drop_duplicates, filter iterate to fixpoint
- [ ] Implement `save_processed()` + `load_processed()`
- [ ] Implement `scripts/run_pipeline.py::main()` — entrypoint CLI
- [ ] Chạy `python scripts/run_pipeline.py` → in ra row counts trước/sau
- [ ] Cập nhật `docs/data-dictionary.md` với số liệu thật
- [ ] Cập nhật `reports/eda_summary.md` (bảng trước/sau filter)

**File cần implement:**
- `src/data_processing.py` — signature đã có, điền TODO
- `scripts/run_pipeline.py` — điền main()
- `docs/data-dictionary.md` — điền số thật
- `reports/eda_summary.md` — điền TBD

**Done khi:** Duong + Loan load được parquet mà không cần raw CSV.

### D4 CN 27/07 — **HỌP REVIEW**
- [ ] Trình bày EDA + parquet + data dictionary
- [ ] Sửa note nếu Lead/QA góp ý

### D5–D9 28/07–01/08 — Buffer + support

Tâm rảnh sau T03:
- [ ] Hỗ trợ Duong/Loan nếu parquet có vấn đề
- [ ] Hỗ trợ Kiên test data pipeline nếu cần

## Phase 2 — Report & Ship (D10–D14)

### D10 T7 02/08 — T12 Final report (section Dataset)

**Sub-task checklist:**
- [ ] Mở `reports/final_report.md` section 2 (Dataset & preprocessing)
- [ ] Điền: schema, stats trước/sau filter, sparsity, distribution
- [ ] Reference `eda_summary.md` cho chi tiết

**File:**
- `reports/final_report.md` — section 2

### D11–D14 — Polish + rehearsal
- [ ] Cross-review report
- [ ] Có mặt rehearsal D14

## Lưu ý kỹ thuật

- `ratings.csv` ~678MB → dtype `int32/float32`, đọc đúng cột cần thiết
- Không tạo dense pivot toàn bộ user × movie
- Nếu clean quá chậm trên 25M rows → sample 5M rows đầu debug, sau đó chạy full

## Done khi (toàn sprint)

Parquet xong D3 26/07 (không block downstream). Final report section 2 xong D11 03/08.
