# Quy ước đặt tên và Git workflow

Tài liệu này là nguồn tham chiếu chính cho cách đặt tên file, branch, commit, Pull Request và artifact trong repo **Hybrid Movie Recommender**.
Nếu phần nào chưa rõ, hãy ưu tiên làm theo ví dụ gần nhất đang có trong repo.

`Hybrid Movie Recommender · AIO Conquer 2026 · Module 02 · Không dùng Jira`

---

## 1. Đặt tên Git branch

`main` là nhánh ổn định. Không push trực tiếp lên `main`; mọi thay đổi nên đi qua branch riêng và Pull Request.

### Định dạng

```text
<type>/<short-kebab-case-description>
```

Ví dụ:

```text
feat/streamlit-three-tabs
data/clean-movies-ratings
ml/content-genre-cosine
test/regression-app-freeze
docs/final-report-outline
```

Không thêm mã định danh ở cuối branch, commit hay PR.

### Type được phép

| Type | Dùng cho | Ví dụ |
| --- | --- | --- |
| `feat` | Tính năng mới, deliverable người dùng thấy được | `feat/streamlit-three-tabs` |
| `fix` | Sửa lỗi | `fix/cold-start-empty-user` |
| `docs` | Chỉ sửa tài liệu | `docs/update-onboarding` |
| `data` | Xử lý dữ liệu, parquet, data dictionary | `data/clean-movies-ratings` |
| `ml` | Thuật toán gợi ý, training, similarity, evaluation logic | `ml/cf-item-similarity` |
| `app` | Streamlit UI và tích hợp app | `app/add-content-tab` |
| `test` | Unit test, smoke test, regression test | `test/recommender-edge-cases` |
| `report` | Báo cáo, chart, kết quả phân tích | `report/final-report` |
| `ci` | CI/CD, workflow, automation | `ci/pytest-workflow` |
| `chore` | Config, cleanup, tooling | `chore/ruff-config` |
| `refactor` | Đổi cấu trúc code, không đổi hành vi | `refactor/recommender-shared-utils` |

### Quy tắc

- Type viết thường; mô tả dùng `kebab-case`.
- Không dùng dấu cách, không dùng camelCase, không dùng tiếng Việt có dấu trong tên branch.
- Mô tả ngắn gọn 2-5 từ, nói về kết quả cần làm.
- Không thêm mã định danh ở cuối tên branch.

---

## 2. Git workflow

```bash
# 0. Luôn bắt đầu từ main mới nhất
git checkout main
git pull origin main

# 1. Tạo branch riêng
git checkout -b ml/cf-item-similarity

# 2. Làm việc và commit theo từng cụm logic
git add src/recommender_cf.py tests/test_recommender.py
git commit -m "ml: add item similarity for collaborative filtering"

# 3. Push branch
git push -u origin ml/cf-item-similarity

# 4. Tạo Pull Request trên GitHub, đợi review và test xanh

# 5. Sau khi merge, cập nhật local và xóa branch đã merge
git checkout main
git pull origin main
git branch -d ml/cf-item-similarity
```

Quy tắc an toàn:

- Không push trực tiếp lên `main`.
- Không `git push --force` lên branch dùng chung.
- Trước khi gửi review, chạy test liên quan và cập nhật branch với `main` mới nhất.
- Tôn trọng ownership trong `README.md`: chỉ sửa file mình own, trừ khi đã thống nhất với owner.

---

## 3. Commit message

Dùng Conventional Commits bản gọn.

### Định dạng

```text
<type>: <tóm tắt dạng mệnh lệnh, viết thường>

[body tùy chọn: giải thích vì sao thay đổi]
```

### Type commit

Dùng cùng bộ type với branch:

| Type | Ý nghĩa | Dùng khi nào | Ví dụ commit |
| --- | --- | --- | --- |
| `feat` | Thêm tính năng mới | Có chức năng mới cho app/demo hoặc workflow người dùng | `feat: add movie search filter` |
| `fix` | Sửa lỗi | Code chạy sai, app lỗi, kết quả recommendation không đúng | `fix: handle empty user ratings` |
| `docs` | Sửa tài liệu | README, onboarding, hướng dẫn, convention, ghi chú setup | `docs: update setup instructions` |
| `data` | Thay đổi xử lý dữ liệu | Load, clean, validate, transform, tạo parquet hoặc data dictionary | `data: clean movies and ratings into parquet` |
| `ml` | Thay đổi thuật toán/model | Simple recommender, content-based, CF, similarity, ranking, metric logic | `ml: add genre cosine recommender` |
| `app` | Thay đổi Streamlit UI | Layout, tab, input, hiển thị kết quả, tích hợp module vào app | `app: integrate recommender tabs` |
| `test` | Thêm hoặc sửa test | Unit test, smoke test, regression test, test cases manual | `test: add recommender edge cases` |
| `report` | Thay đổi báo cáo/kết quả phân tích | Final report, test report, evaluation summary, chart mô tả kết quả | `report: add evaluation summary` |
| `ci` | Thay đổi automation | GitHub Actions, test workflow, lint workflow, script kiểm tra tự động | `ci: run pytest on pull requests` |
| `chore` | Việc phụ trợ, không đổi logic sản phẩm | Config, dependency, cleanup nhỏ, ignore file, tooling | `chore: update requirements` |
| `refactor` | Đổi cấu trúc code không đổi hành vi | Tách hàm, đổi tổ chức module, giảm trùng lặp | `refactor: split recommender helpers` |

### Ví dụ tốt

```text
data: clean movies and ratings into parquet
ml: implement weighted rating recommender
ml: add genre cosine content recommender
ml: exclude seen movies in cf recommendations
app: integrate three recommender tabs
test: add app smoke test for empty data
report: add evaluation summary
docs: update onboarding steps
```

### Quy tắc

- Dùng động từ mệnh lệnh: `add`, `fix`, `update`, `remove`, `refactor`.
- Summary tối đa 72 ký tự, không chấm cuối câu.
- Mỗi commit nên là một thay đổi logic; không gom data processing, app UI và report vào cùng một commit.

---

## 4. Pull Request convention

### Title

```text
<type>: <short summary>
```

Ví dụ:

```text
ml: add collaborative filtering recommendations
app: integrate simple, content and cf tabs
```

### Description nên có

- Thay đổi gì và vì sao cần thay đổi.
- File/module chính bị ảnh hưởng.
- Cách đã test: `pytest tests/ -v`, smoke test Streamlit, hoặc screenshot nếu có UI.
- Rủi ro còn lại nếu chưa test được với dataset thật.

### Review và merge

- Cần ít nhất 1 reviewer approve trước khi merge.
- Reviewer code/app: Tech Leader.
- Reviewer test/bug/regression: QA / Reviewer.
- Ưu tiên **Squash and merge** để lịch sử `main` gọn.
- PR lớn hơn khoảng 400 dòng nên cân nhắc tách nhỏ.

---

## 5. Đặt tên file và folder

| Hạng mục | Quy ước | Ví dụ |
| --- | --- | --- |
| Folder | lowercase, kebab-case hoặc một từ | `data/`, `naming-conventions/`, `plans/roles/` |
| Python module | `snake_case.py` | `data_processing.py`, `recommender_cf.py` |
| Python test | `test_<module>.py` | `test_recommender.py`, `test_data_pipeline.py` |
| Script | `snake_case.py`, bắt đầu bằng động từ nếu chạy trực tiếp | `run_pipeline.py`, `build_cf_artifacts.py` |
| Notebook | `NN_short_snake_case.ipynb` nếu có thứ tự | `01_eda_movies.ipynb`, `05_evaluation_analysis.ipynb` |
| Markdown docs | `kebab-case.md` hoặc tên đã có sẵn | `board-howto.md`, `data-dictionary.md`, `TEAM_BOARD.md` |
| Report | `snake_case.md` | `final_report.md`, `test_report.md` |
| Data raw | giữ tên dataset gốc | `movies.csv`, `ratings.csv` |
| Data processed | `snake_case.parquet` | `movies_clean.parquet`, `ratings_clean.parquet` |
| Artifact | `snake_case` + hậu tố rõ nghĩa | `cf_item_similarity.npz`, `movie_index.pkl` |
| Config | theo default của tool | `requirements.txt`, `.gitignore` |

Quy tắc chung:

- Tên file/folder không dùng dấu cách.
- Code và artifact dùng tiếng Anh không dấu để dễ import/chạy script.
- Tài liệu nội dung có thể viết tiếng Việt; tên file nên giữ ASCII.

---

## 6. Quy ước Python

Theo PEP 8.

| Element | Quy ước | Ví dụ |
| --- | --- | --- |
| Biến / function | `snake_case` | `movie_id`, `load_ratings()` |
| Class | `PascalCase` | `SimpleRecommender`, `ContentRecommender` |
| Constant | `UPPER_SNAKE_CASE` | `DEFAULT_TOP_K`, `MIN_RATINGS` |
| Private helper | bắt đầu bằng `_` | `_normalize_genres()` |
| Module/package | `snake_case` | `src.recommender_cf` |

Quy tắc thêm:

- Function trả về recommendation nên rõ đối tượng: `recommend_top_movies()`, `recommend_similar_movies()`, `recommend_for_user()`.
- Function xử lý data nên bắt đầu bằng động từ: `load_`, `clean_`, `build_`, `save_`, `validate_`.
- Biến DataFrame nên có tên rõ ngữ cảnh: `movies_df`, `ratings_df`, `recommendations_df`.
- Không đặt tên biến trùng với built-in Python: `list`, `dict`, `id`, `type`.

---

## 7. Quy ước data và cột

Dataset chính:

- `movies.csv`
- `ratings.csv`

Tên cột trong data processed nên dùng `snake_case`, lowercase.

| Loại cột | Quy ước | Ví dụ |
| --- | --- | --- |
| ID | giữ dạng `<entity>_id` | `movie_id`, `user_id` |
| Tiêu đề phim | `title` | `title` |
| Genre | `genres` hoặc cột đã tách `genre_*` | `genres`, `genre_action` |
| Rating | tên rõ ngữ cảnh | `rating`, `avg_rating`, `num_ratings` |
| Điểm gợi ý | hậu tố `_score` | `weighted_score`, `similarity_score` |
| Boolean | tiền tố `is_` / `has_` | `is_seen`, `has_genre` |
| Timestamp | hậu tố `_at` | `rated_at`, `processed_at` |

Quy tắc:

- Không dùng tên cột có dấu cách hoặc ký tự đặc biệt.
- ID business theo dataset MovieLens có thể map từ `movieId`, `userId` sang `movie_id`, `user_id` ở layer processed.
- Điểm similarity nên nằm trong khoảng `[0, 1]` nếu thuật toán cho phép.

---

## 8. Recommender naming

| Thành phần | Quy ước | Ví dụ |
| --- | --- | --- |
| Simple recommender module | `recommender_simple.py` | `get_top_movies()` |
| Content-based module | `recommender_content.py` | `recommend_similar_movies()` |
| Collaborative filtering module | `recommender_cf.py` | `recommend_for_user()` |
| Evaluation module | `evaluation.py` | `hit_rate_at_k()`, `ndcg_at_k()` |
| Artifact CF | tiền tố `cf_` | `cf_item_similarity.npz` |
| Model/index artifact | `snake_case` | `movie_id_to_index.pkl` |

Metric naming:

- `hr_at_10`
- `ndcg_at_10`
- `genre_overlap_at_k`
- `coverage`
- `precision_at_k` nếu có

---

## 9. Streamlit app naming

| Hạng mục | Quy ước | Ví dụ |
| --- | --- | --- |
| File app | `src/app/streamlit_app.py` | `streamlit run src/app/streamlit_app.py` |
| Model adapter | `src/app/model_adapter.py` | `predict(user_id, movies, ratings, top_k)` |
| View label | ngắn, đúng domain | `Recommend`, `Context` |
| Session state key | `snake_case` | `selected_movie_id`, `selected_user_id` |
| Cached function | động từ + đối tượng | `load_data()`, `load_hybrid_artifacts()` |

Quy tắc:

- UI label có thể viết tiếng Việt nếu app demo cho lớp/nhóm.
- Key nội bộ của Streamlit nên dùng tiếng Anh không dấu.
- Không hard-code path rải rác; ưu tiên hằng số như `DATA_DIR`, `ARTIFACT_DIR`.

---

## 10. Notebook, report và test

### Notebook

- Dùng prefix số nếu notebook có thứ tự trong flow: `01_`, `02_`, ...
- Mỗi notebook nên có markdown cell đầu tiên nêu mục tiêu và input/output.
- Output lớn không cần commit nếu làm repo nặng.

### Report

- Báo cáo chính: `reports/final_report.md`.
- Báo cáo test: `reports/test_report.md`.
- Báo cáo EDA/evaluation: `snake_case.md`, ví dụ `eda_summary.md`, `cf_evaluation.md`.

### Test

| Loại test | File | Ví dụ case |
| --- | --- | --- |
| Data pipeline | `tests/test_data_pipeline.py` | clean schema, missing values |
| Recommender | `tests/test_recommender.py` | top-k, exclude seen, cold-start |
| App smoke | `tests/test_app_smoke.py` | import app, render basic |
| Test plan | `tests/test_cases.md` | manual/UAT cases |

Chạy trước khi gửi PR:

```bash
pytest tests/ -v
```

---

## 11. Versioning và milestone tag

Dùng tag theo dạng:

```text
vMAJOR.MINOR.PATCH
```

Gợi ý milestone:

```text
v0.1.0  App skeleton + data pipeline
v0.2.0  Simple + content-based recommender
v0.3.0  Collaborative filtering + evaluation
v1.0.0  Final demo + report + tests
```

Tạo tag sau khi milestone đã merge vào `main`:

```bash
git tag -a v0.3.0 -m "Collaborative filtering and evaluation"
git push origin v0.3.0
```

---

## Quick reference

```text
Branch:   ml/cf-item-similarity
Commit:   ml: add item similarity for cf
PR:       ml: add collaborative filtering recommendations
Module:   recommender_cf.py
Test:     test_recommender.py
Notebook: 04_cf_experiments.ipynb
Report:   final_report.md
Artifact: cf_item_similarity.npz
```
