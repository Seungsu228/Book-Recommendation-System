"""MF ranking diagnosis: checks why recommending by predicted rating gives lower Precision@10 than popularity, and compares alternatives.

Run: python notebooks/mf_ranking_diag.py   (trains MF 5 times, about 12 minutes)
Output: results/mf_ranking_comparison.csv (one row per alternative)

Same evaluation conditions as knn_tune.py and mf_dev.py (RMSE on 50,000 val ratings, Precision/Recall on 2,000 users, fixed seed).
The letters (A~G) match the table in the presentation.
  A: by predicted rating (original)    B: A + filter out books with few ratings
  C: P·Q only, no bias                 D: A + α·log(rating count)  (popularity weighting)
  E: sum of Q cosine similarity to liked books (current recommend in mf.py)
  F: retrain with higher reg + A       G: retrain with a separate reg on biases only + A
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # add the repo root folder to the import path

import numpy as np
import pandas as pd
from numba import njit

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.baseline import Popularity
from src.models.mf import MatrixFactorization

SAMPLE_N = 50_000
MAX_USERS = 2_000

train, val, _ = load_data()
book_count = train.groupby("book_id").size()   # number of train ratings per book (= popularity)
print("train ratings per book:", book_count.describe(percentiles=[.01, .1, .5]).round(1).to_dict(), flush=True)


# ── Wrapper that swaps only the ranking rule ─────────────────────────
# Keeps the trained MF as is and replaces only the "score computation" of recommend, for comparison.
# score_fn(model, uid) → array of scores for the 10,000 books. Excluding read books and picking the top k work as in mf.py.
class Ranker:
    def __init__(self, model, score_fn):
        self.model, self.score_fn = model, score_fn

    def recommend(self, user_id, k=10):
        m = self.model
        if user_id not in m.user_to_idx_:            # unseen users follow the original model's handling (popularity)
            return m.recommend(user_id, k)
        scores = np.array(self.score_fn(m, m.user_to_idx_[user_id]), dtype=np.float64)
        scores[[m.item_to_idx_[b] for b in m.seen_[user_id]]] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k]
        return m.item_ids_[top[np.argsort(-scores[top])]].tolist()


def predicted(m, uid):
    """A: predicted rating (same formula as predict, no clipping). global_mean and b_u are constant per user, so they do not affect the ranking."""
    return m.global_mean_ + m.user_bias_[uid] + m.item_bias_ + m.Q_ @ m.P_[uid]


rows = []


def evaluate(model, name, rmse):
    """Records Precision/Recall@K, and also how popular the recommended books are."""
    res = {"method": name, "rmse": rmse}
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    # How many train ratings the recommended books (300 users x 10 books) received
    #   rec_count_median: median (over 10,000 for popularity, hundreds for mostly niche books)
    #   rec_count_lt100 : share of books with fewer than 100 ratings
    users = np.random.default_rng(1).choice(val["user_id"].unique(), 300, replace=False)
    recs = np.concatenate([model.recommend(int(u), 10) for u in users])
    c = book_count.reindex(recs).to_numpy()
    res["rec_count_median"] = float(np.median(c))
    res["rec_count_lt100"] = float((c < 100).mean())
    rows.append(res)
    print(f"  {name:<40} rmse={rmse:.4f} p@10={res['precision@10']:.4f} "
          f"median ratings of recommended books={res['rec_count_median']:.0f}", flush=True)


def rmse_of(model):
    return evaluate_rating(model, val, sample_n=SAMPLE_N)["rmse"]


# ── Reference: popularity ────────────────────────────────────────────
pop = Popularity().fit(train)
evaluate(pop, "Popularity", rmse_of(pop))

# ── Train the default MF + diagnose the cause ────────────────────────
t0 = time.time()
base = MatrixFactorization().fit(train)
print(f"default MF training {time.time() - t0:.0f}s", flush=True)
base_rmse = rmse_of(base)   # A~E only change the ranking on the same model, so they all have this RMSE
cnt = book_count.reindex(base.item_ids_).to_numpy()   # rating counts in matrix-position order

print("\n== Diagnosis ==")
# (1) Check the hypothesis "books with few ratings get extreme item biases": correlation of bias and rating count
print("corr(item bias, log rating count):", round(np.corrcoef(base.item_bias_, np.log(cnt))[0, 1], 3))
for lo, hi in [(0, 100), (100, 300), (300, 1000), (1000, 10**6)]:
    mk = (cnt >= lo) & (cnt < hi)
    print(f"  rating count {lo}~{hi}: {mk.sum()} books, mean item bias {base.item_bias_[mk].mean():+.3f}, std {base.item_bias_[mk].std():.3f}")

# (2) Books with the largest bias: extreme because of few ratings, or truly high average?
titles = pd.read_csv(ROOT / "data" / "books.csv").set_index("book_id")["title"]
book_mean = train.groupby("book_id")["rating"].mean()
print("top 5 books by item bias (rating count, bias, actual mean):")
for i in np.argsort(-base.item_bias_)[:5]:
    b = base.item_ids_[i]
    print(f"  {cnt[i]:>5}  {base.item_bias_[i]:+.3f}  {book_mean[b]:.2f}  {titles[b]}")

# (3) In the predicted rating, which part drives the ranking: the "same for everyone" part (b_i) or the "personal" part (P·Q)?
u0 = base.user_to_idx_[int(val["user_id"].iloc[0])]
print(f"score spread across books (std): item bias {base.item_bias_.std():.3f} vs P·Q {(base.Q_ @ base.P_[u0]).std():.3f}"
      "  → if bias dominates, every user gets a similar list")

# ── Change only the ranking rule (no retraining, same RMSE) ──────────────
print("\n== Ranking rule comparison ==")
evaluate(Ranker(base, predicted), "A. predicted rating (current)", base_rmse)
for n in [100, 300, 1000]:
    # books with fewer than n ratings get -inf and are removed from the candidates
    mask = np.where(cnt >= n, 0.0, -np.inf)
    evaluate(Ranker(base, lambda m, u, mask=mask: predicted(m, u) + mask),
             f"B. predicted, books with >= {n} ratings", base_rmse)
evaluate(Ranker(base, lambda m, u: m.Q_ @ m.P_[u]), "C. P.Q only (no bias)", base_rmse)
log_cnt = np.log(cnt)
for a in [0.1, 0.3, 1.0]:
    # more popular books get a higher score; the larger α is, the closer to the popularity ranking.
    evaluate(Ranker(base, lambda m, u, a=a: predicted(m, u) + a * log_cnt),
             f"D. predicted + {a}*log(count)", base_rmse)
evaluate(base, "E. sum cos(Q_i, Q_liked) (KNN-style)", base_rmse)   # the recommend in mf.py as is

# ── Alternatives that need retraining ─────────────────────────────────
print("\n== Retraining alternatives ==")
for reg in [0.05, 0.1]:
    # F: would a higher reg shrink the biases? → P·Q shrink too, so the bias share actually grows
    m = MatrixFactorization(reg=reg).fit(train)
    evaluate(Ranker(m, predicted), f"F. reg={reg}, predicted", rmse_of(m))


# G: SGD with a separate (stronger) reg on the biases only. Same as _sgd_epoch in mf.py, except the bias update uses reg_bias instead of reg.
#    An experimental variant not added to mf.py, so it lives only here.
@njit(cache=True)
def _sgd_epoch_bias_reg(order, rows, cols, ratings, global_mean,
                        user_bias, item_bias, P, Q, lr, reg, reg_bias):
    n_factors = P.shape[1]
    for pos in order:
        u = rows[pos]
        i = cols[pos]
        dot = 0.0
        for f in range(n_factors):
            dot += P[u, f] * Q[i, f]
        err = ratings[pos] - (global_mean + user_bias[u] + item_bias[i] + dot)
        user_bias[u] += lr * (err - reg_bias * user_bias[u])   # ← only here
        item_bias[i] += lr * (err - reg_bias * item_bias[i])   # ← and here use reg_bias
        for f in range(n_factors):
            pu = P[u, f]
            qi = Q[i, f]
            P[u, f] += lr * (err * qi - reg * pu)
            Q[i, f] += lr * (err * pu - reg * qi)


def fit_bias_reg(reg_bias):
    # fit with n_epochs=0 builds only the ID mapping, means and initial P/Q, without training.
    m = MatrixFactorization(n_epochs=0).fit(train)
    # To reproduce the random sequence of fit in mf.py: start from the same seed and draw (and discard) the numbers used to initialize P and Q.
    # Then the shuffle order of each epoch matches the default MF, giving a fair comparison where "only reg_bias differs".
    rng = np.random.default_rng(m.seed)
    rng.normal(size=(m.n_users, m.n_factors))
    rng.normal(size=(m.n_items, m.n_factors))
    rows_ = train["user_id"].map(m.user_to_idx_).to_numpy(np.int64)
    cols_ = train["book_id"].map(m.item_to_idx_).to_numpy(np.int64)
    ratings = train["rating"].to_numpy(np.float64)
    for _ in range(20):
        _sgd_epoch_bias_reg(rng.permutation(len(rows_)), rows_, cols_, ratings, m.global_mean_,
                            m.user_bias_, m.item_bias_, m.P_, m.Q_, m.lr, m.reg, reg_bias)
    return m


for rb in [0.5, 2.0]:
    m = fit_bias_reg(rb)
    evaluate(Ranker(m, predicted), f"G. reg_bias={rb}, predicted", rmse_of(m))

# ── Save ───────────────────────────────────────────────────────────
out = pd.DataFrame(rows)[["method", "rmse", "precision@5", "recall@5", "precision@10",
                          "recall@10", "rec_count_median", "rec_count_lt100"]]
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results" / "mf_ranking_comparison.csv", index=False, float_format="%.4f")
print("\nsaved: results/mf_ranking_comparison.csv")
print(out.round(4).to_string(index=False))
