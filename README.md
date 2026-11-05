# Goodbooks Recommender

Book recommendation system on Goodbooks-10k using collaborative, content-based, and hybrid filtering.
Team 7, Machine Learning course (2026).

## Overview

The system predicts how a reader would rate unread books and recommends the Top-N books,
combining Goodreads user ratings with book tags and author information.

We implement and compare four filtering methods under the same data split and metrics:

| Method | Model | File |
| --- | --- | --- |
| Collaborative (memory-based) | Item-based KNN | `src/models/knn.py` |
| Collaborative (model-based) | Matrix Factorization (SGD, L2 regularization) | `src/models/mf.py` |
| Content-based | TF-IDF on tags, authors, titles + cosine similarity | `src/models/content.py` |
| Hybrid | Weighted blend of CF and content scores | `src/models/hybrid.py` |

Baselines (global mean, book mean, popularity) are in `src/models/baseline.py`.

## Dataset

[Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k): 10,000 books, 53,424 users, about 6 million ratings (1–5).
The raw data is not included in this repository. See [`data/README.md`](data/README.md) for download instructions.

## Project Structure

```
goodbooks-recsys/
├── data/            # raw data (not tracked, see data/README.md)
├── notebooks/       # EDA and experiments
├── results/         # figures and result tables
└── src/
    ├── preprocess.py    # loading, preprocessing, train/val/test split
    ├── evaluate.py      # RMSE, MAE, Precision@K, Recall@K
    └── models/
        ├── base.py      # common Recommender interface
        ├── baseline.py
        ├── knn.py
        ├── mf.py
        ├── content.py
        └── hybrid.py
```

All models inherit from `Recommender` in `src/models/base.py` and implement:

- `fit(train)`: train on rating data
- `predict(user_id, book_id)`: predicted rating (1–5)
- `recommend(user_id, k)`: Top-k unrated book IDs

## Getting Started

```bash
git clone <this-repo-url>
cd goodbooks-recsys
pip install -r requirements.txt
# download the dataset into data/ (see data/README.md)
```

## Evaluation

- Split: per-user random hold-out into train / validation / test (seed 42)
- Rating prediction: RMSE, MAE
- Top-N: Precision@K, Recall@K (K = 5, 10), ratings ≥ 4 treated as relevant
- Cold-start: most ratings of selected books are hidden to compare methods
- Hyperparameters are tuned on the validation set only; the test set is used once at the end

## Results

_To be added._

## Contributing (team workflow)

1. Never push directly to `main`.
2. Create a branch per task: `feature/<name>` (e.g. `feature/knn`).
3. Open a pull request; at least one teammate reviews before merging.
4. Cite any external code (including AI-generated code) in comments, and explain
   functions and parameters not covered in class.

## Team


## License

[MIT](LICENSE)
