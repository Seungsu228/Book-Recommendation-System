"""MatrixFactorization check script: compares it with BookMean, Popularity and ItemKNN on val.

Run: python notebooks/mf_dev.py
Same evaluation conditions as notebooks/knn_tune.py (RMSE on 50,000 val ratings, Precision/Recall on 2,000 users, fixed seed).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # add the repo root folder to the import path

import numpy as np
import pandas as pd

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.baseline import BookMean, Popularity
from src.models.knn import ItemKNN
from src.models.mf import MatrixFactorization

SAMPLE_N = 50_000   # number of val ratings for RMSE/MAE
MAX_USERS = 2_000   # number of val users for Precision/Recall@K

train, val, _ = load_data()   # compare on val only (test is used once, after choosing the final model)
books = pd.read_csv(ROOT / "data" / "books.csv").set_index("book_id")["title"]
book_count = train.groupby("book_id").size()   # number of train ratings per book (= popularity)

# ── 1. Training ──────────────────────────────────────────────────
# ItemKNN uses the setting chosen in knn_tune.py (pearson, k=40).
# MF uses the setting chosen in mf_tune.py (n_factors=100, reg=0.02).
#   Going from n_factors 50 → 100, RMSE got worse by only 0.002 while Precision@10 rose from 0.052 → 0.072 (results/mf_tuning.csv).
#   The default in mf.py (n_factors=50) is left as is; the setting is given only here.
models = {
    "BookMean": BookMean(),
    "Popularity": Popularity(),
    "ItemKNN (pearson, k=40)": ItemKNN(k=40, similarity="pearson"),
    "MF": MatrixFactorization(n_factors=100, reg=0.02),
}
for name, m in models.items():
    t0 = time.time()
    m.fit(train)
    print(f"{name} fit: {time.time() - t0:.1f}s", flush=True)

# ── 2. Check that MF works ────────────────────────────────────────
mf = models["MF"]
row = val.iloc[0]
print("\nfirst val row:", int(row.user_id), int(row.book_id), "actual", row.rating,
      "→ MF predicted", round(mf.predict(row.user_id, row.book_id), 3))
print("unseen user/book:", round(mf.predict(10**9, 2), 3),
      round(mf.predict(int(row.user_id), 10**9), 3))

u = int(row.user_id)
recs = mf.recommend(u, 5)
print(f"recommendations for user {u}:", recs, "/ overlap with books already read:", len(set(recs) & mf.seen_[u]))
for b in recs:
    # Also show how many people rated each recommended book in train (to check whether only unpopular books come up)
    print(f"    {book_count[b]:>5} ratings  {books[b]}")

# ── 3. Evaluation ────────────────────────────────────────────────
# Every model is evaluated on the same 50,000 ratings and 2,000 users (fixed seed).
rows = {}
for name, m in models.items():
    t0 = time.time()
    res = evaluate_rating(m, val, sample_n=SAMPLE_N)
    res.update(evaluate_topk(m, val, max_users=MAX_USERS))
    rows[name] = res
    print(f"{name} eval: {time.time() - t0:.1f}s", flush=True)

result = pd.DataFrame(rows).T
print("\nval results")
print(result.round(4).to_string())

# ── 4. Precision@10: is MF above popularity? ───────────────────────
p_mf = result.loc["MF", "precision@10"]
p_pop = result.loc["Popularity", "precision@10"]
print(f"\nPrecision@10  MF {p_mf:.4f} vs Popularity {p_pop:.4f} →",
      "MF is higher" if p_mf > p_pop else "MF is lower")

# How much the lists lean toward "popular books": median train rating count of the recommended books
# (thousands for the popularity list; tens to hundreds if mostly lesser-known books)
users = np.random.default_rng(1).choice(val["user_id"].unique(), 300, replace=False)
print("median train rating count of recommended books (300 users x 10 books):")
for name in ["Popularity", "ItemKNN (pearson, k=40)", "MF"]:
    recs = np.concatenate([models[name].recommend(int(u), 10) for u in users])
    print(f"  {name:<25} {np.median(book_count.reindex(recs)):.0f}")
