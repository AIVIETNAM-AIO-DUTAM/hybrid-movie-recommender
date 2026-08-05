# Raw dataset

Raw dataset dùng cho project là Kaggle Movie Recommendation System
theo schema MovieLens-style (`movies.csv`, `ratings.csv`):

| File | Size | Schema |
|------|------|--------|
| `movies.csv` | ~2.9 MB | `movieId,title,genres` |
| `ratings.csv` | ~647 MB | `userId,movieId,rating,timestamp` |

Nguồn: https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=movies.csv

> CSV lớn **không** commit git (đã `.gitignore`).

Tiếp theo — Data Engineer (Trần Hoàng Minh Tâm):

```bash
source .venv/bin/activate
# EDA notebooks, rồi:
python -m src.data_processing
```
