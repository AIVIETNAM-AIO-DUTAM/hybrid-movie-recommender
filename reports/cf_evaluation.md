# CF Evaluation

Owner: ML B · Status: draft

| Metric | Value | Sample size | Notes |
|--------|-------|-------------|-------|
| HR@10 | TBD | | leave-last-out by timestamp |
| NDCG@10 | TBD | | |

## Reproduce

```bash
source .venv/bin/activate
python -c "
import sys; sys.path.insert(0, 'src')
import pandas as pd
from data_processing import load_processed
from evaluation import run_evaluation
movies, ratings = load_processed()
scores = run_evaluation(ratings, movies, sample_size=200)
print(scores.to_string(index=False))
"
```

`run_evaluation()` tự động:
1. leave-last-out split theo timestamp (per-user)
2. build CF trên train
3. sample 200 user, recommend top-10, check HR@10 / NDCG@10

Số liệu ghi vào bảng trên sau khi chạy xong.
