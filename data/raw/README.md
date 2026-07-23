# Raw dataset

Đã tải sẵn (MovieLens 25M — khớp `document.pdf`):

| File | Size | Schema |
|------|------|--------|
| `movies.csv` | ~2.9 MB | `movieId,title,genres` |
| `ratings.csv` | ~647 MB | `userId,movieId,rating,timestamp` |

Nguồn: https://files.grouplens.org/datasets/movielens/ml-25m.zip

> CSV lớn **không** commit git (đã `.gitignore`).

Tiếp theo — Data Engineer (Trần Hoàng Minh Tâm):

```bash
source .venv/bin/activate
# EDA notebooks, rồi:
python -m src.data_processing
```
