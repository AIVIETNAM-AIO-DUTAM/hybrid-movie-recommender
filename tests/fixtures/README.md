# Test Data And Model Fixtures

Folder này chứa bộ dữ liệu và model nhỏ để tester chạy app/tests mà không cần
commit data/model full.

Các file được tạo bằng:

```bash
python tests/fixtures/build_test_assets.py
```

Chạy Streamlit bằng fixture:

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
