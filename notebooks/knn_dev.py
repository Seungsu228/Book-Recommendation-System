import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add the repo root folder to the import path

from src.preprocess import load_data
import numpy as np
from src.models.knn import ItemKNN

train, val, test = load_data()
t0 = time.time()
m = ItemKNN().fit(train)
print(f"fit time: {time.time() - t0:.1f}s")

# ── Check steps 1~2 ──
print("R_ shape:", m.R_.shape)            # (number of users, number of books)
print("R_ nnz:", m.R_.nnz, "/ train:", len(train))  # the two values should be equal
u, b = train.iloc[0]["user_id"], train.iloc[0]["book_id"]
print("example:", int(u), int(b), "rating", train.iloc[0]["rating"], "→ R_ value", m.R_[m.uidx_[u], m.iidx_[b]])

# ── Check step 3 ──
print("global mean:", round(m.global_mean_, 4), "/ computed with pandas:", round(train["rating"].mean(), 4))
b_mean = train.groupby("book_id")["rating"].mean()
print("book", int(b), "mean:", round(float(m.item_mean_[m.iidx_[b]]), 4), "/ computed with pandas:", round(b_mean[b], 4))
print("3 most popular books:", m.item_ids_[m.popular_[:3]].tolist())

# ── Check step 4 ──
import pandas as pd
books = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "books.csv").set_index("book_id")["title"]
print("stored neighbors:", m.S_.nnz, "(at most 10,000 books x k=20)")
q = 2  # Harry Potter and the Sorcerer's Stone
row = m.S_[m.iidx_[q]]
order = np.argsort(-row.data)[:5]
print(f"books similar to '{books[q]}':")
for j, s in zip(row.indices[order], row.data[order]):
    print(f"  {s:.3f}  {books[m.item_ids_[j]]}")

# ── Check step 5 ──
from src.evaluate import evaluate_rating
from src.models.baseline import BookMean
row = val.iloc[0]
print("first val row:", int(row.user_id), int(row.book_id), "actual", row.rating, "→ predicted", round(m.predict(row.user_id, row.book_id), 3))
print("unseen user/book:", round(m.predict(10**9, 2), 3), round(m.predict(int(row.user_id), 10**9), 3))
print("RMSE/MAE (50,000 val ratings)")
print("  BookMean:", evaluate_rating(BookMean().fit(train), val, sample_n=50000))
print("  ItemKNN :", evaluate_rating(m, val, sample_n=50000))

# ── Check step 6 ──
from src.evaluate import evaluate_topk
from src.models.baseline import Popularity

u = int(val.iloc[0].user_id)
recs = m.recommend(u, 5)
print(f"recommendations for user {u}:", recs, "/ overlap with books already read:", len(set(recs) & set(train[train.user_id == u].book_id)))
for b2 in recs:
    print("   ", books[b2])

# For comparison: what happens if we recommend by predicted rating (predict)
def recommend_by_predict(model, user_id, k=10):
    uid = model.uidx_[user_id]
    rated = model.R_[uid].indices
    cand = np.setdiff1d(np.arange(len(model.item_ids_)), rated)
    preds = np.array([model.predict(user_id, model.item_ids_[i]) for i in cand])
    return model.item_ids_[cand[np.argsort(-preds)[:k]]].tolist()

class ByPredict:
    def __init__(self, model): self.model = model
    def recommend(self, user_id, k=10): return recommend_by_predict(self.model, user_id, k)

print("Precision/Recall@K (val, 2,000 users)")
print("  Popularity       :", evaluate_topk(Popularity().fit(train), val, max_users=2000))
print("  KNN, by predicted rating (slow, 50 users):", evaluate_topk(ByPredict(m), val, max_users=50))
print("  KNN, by similarity sum:", evaluate_topk(m, val, max_users=2000))
