# Test Data And Model Fixtures

Folder này chứa bộ dữ liệu và model nhỏ để tester chạy app/tests mà không cần
commit data/model full.

> **Note**: `src/app.py` là official demo của sprint (có behavioral test
> `tests/test_app_smoke.py`). `src/app/streamlit_app.py` (MicroLens UI) là
> parallel UI — không có behavioral test, chỉ có AST parse guard. Chạy bằng
> fixtures bên dưới để smoke-test path MicroLens khi cần.

Các file được tạo bằng:

```bash
python tests/fixtures/build_test_assets.py
```

Chạy MicroLens UI (`src/app/streamlit_app.py`) bằng fixture:

```bash
REC_DATA_DIR=tests/fixtures/data/processed \
REC_MODEL_DIR=tests/fixtures/model \
streamlit run src/app/streamlit_app.py
```

Nếu không set `REC_DATA_DIR` và `REC_MODEL_DIR`, app vẫn dùng dữ liệu thật:

```text
data/processed/
model/
```

Lưu ý: `tests/fixtures/data/processed/*.parquet` được lưu qua Git LFS. Nếu clone
mà chưa có LFS, chạy `git lfs install && git lfs pull`, hoặc regenerate bằng
`python tests/fixtures/build_test_assets.py`.
