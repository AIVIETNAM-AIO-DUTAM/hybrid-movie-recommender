# Hybrid Movie Recommender System

> **End-to-end movie recommendation prototype.** Project xây dựng hệ thống gợi ý phim từ MovieLens 25M, gồm preprocessing, recommender models, hybrid scoring, evaluation, automated tests và Streamlit demo. Mục tiêu là chứng minh một luồng recommender hoàn chỉnh: **raw data -> processed parquet -> model artifacts -> hybrid recommendations -> app demo -> QA evidence**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-data%20processing-150458">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-KNN%20%2B%20TF--IDF-F7931E">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-sparse%20matrix-8CAAE6">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-demo-FF4B4B">
  <img alt="Pytest" src="https://img.shields.io/badge/Pytest-46%20passed-0A9EDC">
</p>

`AIO Conquer 2026 · Module 02 · Hybrid Movie Recommender`

---

## 1. Problem

Người xem phim có quá nhiều lựa chọn, nhưng không phải lúc nào cũng biết phim nào phù hợp với sở thích của mình. Project này xây dựng một recommender system để hỗ trợ ba tình huống:

| Use case | Cách xử lý | Ý nghĩa |
|---|---|---|
| User mới hoặc thiếu lịch sử rating | **Simple Recommender** | Gợi ý phim phổ biến bằng weighted rating. |
| Người dùng thích một phim và muốn tìm phim tương tự | **Content-based Recommender** | Tìm phim gần nhau theo nội dung như title/genres. |
| User đã có lịch sử rating | **Collaborative Filtering + Hybrid** | Cá nhân hóa gợi ý dựa trên hành vi rating và nội dung phim. |

**Scope:** MovieLens 25M -> preprocessing -> KNN CF/content artifacts -> hybrid recommendation -> Streamlit demo.

**Out of scope:** production serving, realtime feedback loop, user authentication, online learning, A/B testing thật.

---

## 2. Current Solution

Project hiện có bốn lớp gợi ý:

1. **Simple Recommender**
   - File chính: `src/recommender_simple.py`
   - Tính `avg_rating`, `num_ratings`, `weighted_rating`
   - Phù hợp cho cold-start hoặc fallback.

2. **Content-based Recommender**
   - File chính: `src/recommender_content.py`
   - Dùng genres/title features để tìm phim tương tự.
   - Bản artifact hiện tại dùng TF-IDF + KNN trong `src/ml/train_knn_content.py`.

3. **Collaborative Filtering**
   - File chính: `src/recommender_cf.py`, `src/ml/train_knn_cf.py`
   - Dùng item-based KNN trên sparse movie-user matrix.
   - Loại phim user đã xem khỏi kết quả recommendation.

4. **Hybrid Recommender**
   - File chính: `src/ml/hybrid_rcm.py`, `src/app/model_adapter.py`
   - Kết hợp điểm CF và content:

```text
hybrid_score = alpha * cf_score + (1 - alpha) * content_score
```

Trong app hiện tại, `alpha = 0.8`, tức hệ thống ưu tiên tín hiệu collaborative filtering và dùng content score để bổ trợ.

---

## 3. Architecture

```text
MovieLens CSV
    -> src/data_processing.py
    -> data/processed/*.parquet
    -> scripts/build_hybrid_artifacts.py
    -> model/knn_cf + model/knn_content (generated locally, not committed)
    -> src/app/model_adapter.py
    -> src/app/streamlit_app.py
    -> Top-K recommendations
```

### Runtime flow

```text
User selects userId
    -> app loads processed data and model artifacts
    -> CF candidate scores
    -> content candidate scores
    -> score normalization
    -> hybrid score
    -> remove seen movies
    -> display ranked recommendations
```

---

## 4. Tech Stack

| Layer | Tool | Responsibility |
|---|---|---|
| Data processing | `pandas`, `pyarrow` | Load CSV, clean movies/ratings, save parquet. |
| Sparse modeling | `scipy.sparse` | Store large movie-user and feature matrices efficiently. |
| ML / retrieval | `scikit-learn` | KNN CF, KNN content, TF-IDF vectorization. |
| Artifact storage | `joblib`, `.npz`, `.parquet` | Save trained models, mappings, sparse matrices. |
| App | `Streamlit` | MicroLens Workbench demo. |
| QA | `pytest` | Unit tests, data fixture tests, app adapter smoke tests. |
| Analysis | `notebooks/`, `evaluation/` | EDA, modeling experiments, hybrid metrics. |

---

## 5. Repository Structure

```text
.
├── data/
│   ├── raw/                     # movies.csv, ratings.csv (not committed)
│   └── processed/               # movies_clean, ratings_cf, ratings_content, train/test parquet
├── docs/
│   ├── data-dictionary.md
│   ├── onboarding.md
│   └── naming-conventions/
├── evaluation/
│   ├── hybrid/                  # alpha comparison and per-user hybrid metrics
│   ├── cf_user_metrics.csv
│   ├── content_user_metrics.csv
│   └── model_comparison.csv
├── model/
│   ├── knn_cf/                  # generated locally by scripts/build_hybrid_artifacts.py
│   └── knn_content/             # generated locally by scripts/build_hybrid_artifacts.py
├── notebooks/
│   ├── 01_eda_movies.ipynb
│   ├── 02_eda_ratings.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_cf_experiments.ipynb
│   └── 05_evaluation_analysis.ipynb
├── plans/
│   └── roles/                   # role-level task planning
├── reports/
│   ├── eda_summary.md
│   ├── cf_evaluation.md
│   ├── test_report.md
│   └── final_report.md
├── scripts/
│   ├── build_hybrid_artifacts.py
│   └── run_hybrid_evaluation.py
├── src/
│   ├── data_processing.py
│   ├── recommender_simple.py
│   ├── recommender_content.py
│   ├── recommender_cf.py
│   ├── evaluation.py
│   ├── app.py                   # legacy 3-tab demo
│   ├── app/
│   │   ├── model_adapter.py
│   │   └── streamlit_app.py      # current MicroLens Workbench
│   └── ml/
│       ├── train_knn_cf.py
│       ├── train_knn_content.py
│       ├── hybrid_rcm.py
│       └── evalu_hybid.py
├── tests/
│   ├── fixtures/                # small committed data/model fixtures
│   ├── test_recommender.py
│   ├── test_data_pipeline.py
│   └── test_app_smoke.py
├── TEAM_BOARD.md                # root-level project board kept for visibility
├── document.pdf                 # root-level source spec kept for visibility
├── requirements.txt
└── README.md
```

Notes:

- `model/` is ignored by git and is **not expected in a fresh clone**. Create it by running `python scripts/build_hybrid_artifacts.py`.
- `tests/fixtures/model/` is committed separately as a tiny QA fixture bundle.
- `TEAM_BOARD.md` and `document.pdf` stay at the repository root intentionally because they are project-level reference files; supporting docs live in `docs/`, planning files in `plans/`, and deliverable reports in `reports/`.

---

## 6. Quickstart

### 6.1 Setup environment

```bash
git clone <repo-url>
cd hybrid-movie-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 6.2 Prepare raw data

Download the Kaggle movie recommendation dataset and place the CSV files here:

```text
data/raw/movies.csv
data/raw/ratings.csv
```

Dataset source:

```text
https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=movies.csv
```

The project expects the MovieLens-style files `movies.csv` and `ratings.csv`
with the schema documented in `docs/data-dictionary.md`.

Raw CSV files are large and should not be committed.

### 6.3 Run preprocessing

```bash
python -m src.data_processing
```

Expected processed outputs:

```text
data/processed/movies_clean.parquet
data/processed/ratings_cf.parquet
data/processed/ratings_content.parquet
```

`src/ml/train_knn_cf.py` will also create:

```text
data/processed/rating_cf_train.parquet
data/processed/rating_cf_test.parquet
```

### 6.4 Build hybrid artifacts

```bash
python scripts/build_hybrid_artifacts.py
```

Expected artifact folders:

```text
model/knn_cf/
model/knn_content/
```

These folders are generated outputs and are ignored by git. If they are missing after clone, run the build command above.

Important files:

```text
model/knn_cf/knn_cf_model.joblib
model/knn_cf/movie_user_matrix.npz
model/knn_cf/cf_mappings.joblib
model/knn_content/knn_content_model.joblib
model/knn_content/movie_feature_matrix.npz
model/knn_content/content_mappings.joblib
model/knn_content/tfidf_vectorizer.joblib
```

### 6.5 Run Streamlit demo

Current app:

```bash
streamlit run src/app/streamlit_app.py
```

The current app is **MicroLens Workbench**. It loads:

```text
data/processed/
model/knn_cf/
model/knn_content/
```

Legacy 3-tab demo, kept for reference:

```bash
streamlit run src/app.py
```

---

## 7. Run With Small Test Fixtures

The repository includes a small committed fixture bundle so QA can run tests or demo logic without full MovieLens 25M.

```bash
REC_DATA_DIR=tests/fixtures/data/processed \
REC_MODEL_DIR=tests/fixtures/model \
streamlit run src/app/streamlit_app.py
```

Fixture folders:

```text
tests/fixtures/data/processed/
tests/fixtures/model/
```

Regenerate fixtures:

```bash
python tests/fixtures/build_test_assets.py
```

---

## 8. Evaluation

Run hybrid evaluation:

```bash
python scripts/run_hybrid_evaluation.py
```

Outputs are saved in:

```text
evaluation/hybrid/
```

Current stored hybrid comparison:

| Alpha | precision@10 | recall@10 | hit_rate@10 | Evaluated users |
|---:|---:|---:|---:|---:|
| 0.5 | 0.038 | 0.039 | 0.22 | 100 |
| 0.6 | 0.063 | 0.084 | 0.36 | 100 |
| 0.7 | 0.075 | 0.099 | 0.44 | 100 |
| 0.8 | **0.076** | **0.105** | **0.44** | 100 |
| 0.9 | 0.073 | 0.104 | 0.44 | 100 |

The app default uses `alpha = 0.8`.

---

## 9. Testing

Run all tests:

```bash
pytest tests -q -rs -p no:cacheprovider
```

Current test suite:

| File | Coverage | Current status |
|---|---|---|
| `tests/test_recommender.py` | Simple, content-based, CF, eval helpers | 10 passed |
| `tests/test_data_pipeline.py` | Processed fixture schema, year, dtype, rating range, duplicates | 3 passed |
| `tests/test_app_smoke.py` | App adapter prediction and Streamlit layout import | 2 passed |
| `tests/test_model_adapter.py` | Hybrid adapter contract, fallback paths | 3 passed |
| `tests/test_ml_recommenders.py` | ML recommender integration behavior | passed |
| `tests/test_edge_guards.py` | Regression guards for edge cases | passed |

Latest verified result:

```text
46 passed, 3 warnings
```

The warnings come from `joblib` / `numpy` while loading fixture artifacts. They do not fail the tests, but should be monitored when upgrading dependencies.

---

## 10. Ownership

| Role | Owner | Main files |
|---|---|---|
| Team Leader | Tân Dư | `src/app/streamlit_app.py`, `src/app/model_adapter.py`, `TEAM_BOARD.md`, docs/integration |
| Data | Trần Hoàng Minh Tâm | `src/data_processing.py`, `notebooks/01_*`, `notebooks/02_*`, `data/` |
| Model | tran Duong | `src/recommender_simple.py`, `src/recommender_content.py`, recommender design |
| Pipeline | 18- Thanh Loan | `src/recommender_cf.py`, `src/ml/`, `scripts/`, `model/`, `evaluation/` |
| Tester | Hoàng Đức Kiên | `tests/`, `tests/fixtures/`, `reports/test_report.md` |

Shared files can be read by everyone. Changes should be coordinated through `TEAM_BOARD.md` to avoid editing the same file at the same time.

---

## 11. Known Limitations

- The app smoke tests currently verify adapter behavior and module import, not full browser-based UI interaction.
- Content-based recommendation mostly depends on title and genre text; it does not yet use tags, overview, cast, posters, or embeddings.
- Hybrid evaluation currently reports a fixed stored run over 100 users; results may change if the split or user sample changes.
- MovieLens 25M is sparse and large, so full artifact rebuild may be slow on low-memory machines.
- This is a prototype/demo, not a production recommender service.

---

## 12. Useful Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data
python -m src.data_processing

# Build models
python scripts/build_hybrid_artifacts.py

# Evaluate hybrid
python scripts/run_hybrid_evaluation.py

# Run current app
streamlit run src/app/streamlit_app.py

# Run current app with fixtures
REC_DATA_DIR=tests/fixtures/data/processed \
REC_MODEL_DIR=tests/fixtures/model \
streamlit run src/app/streamlit_app.py

# Test
pytest tests -q -rs -p no:cacheprovider
```
