import numpy as np
import pandas as pd
from numba import njit

from .base import Recommender


# ── One SGD epoch (compiled with numba) ─────────────────────────────
# A plain Python for loop is slow because the interpreter checks types and creates objects on every step.
# 4.78 million ratings x 20 epochs = about 100 million steps, which takes over 30 minutes from that overhead alone.
# @njit (numba) compiles this function to machine code the first time it is called, so it runs at about
# the speed of a loop written in C.
# (In nopython mode it can only handle numpy arrays and numbers, so it takes arrays as arguments instead of self.)
# cache=True: saves the compiled code in __pycache__, so later runs skip the compile time (a few seconds).
@njit(cache=True)
def _sgd_epoch(order, rows, cols, ratings, global_mean,
               user_bias, item_bias, P, Q, lr, reg):
    """Go through the ratings in `order` one by one and update b_u, b_i, P_u, Q_i (arrays are modified in place)."""
    n_factors = P.shape[1]
    for pos in order:
        u = rows[pos]
        i = cols[pos]

        # prediction = global mean + b_u + b_i + P_u·Q_i  (the dot product self.P_[u] @ self.Q_[i] written as a loop)
        dot = 0.0
        for f in range(n_factors):
            dot += P[u, f] * Q[i, f]
        err = ratings[pos] - (global_mean + user_bias[u] + item_bias[i] + dot)

        # bias update: b ← b + lr·(err − reg·b)
        user_bias[u] += lr * (err - reg * user_bias[u])
        item_bias[i] += lr * (err - reg * item_bias[i])

        # latent vector update: P_u ← P_u + lr·(err·Q_i − reg·P_u), Q_i ← Q_i + lr·(err·P_u − reg·Q_i)
        # For the same reason the earlier version used .copy() on p_u and q_i, we read the old values (pu, qi) first,
        # so that Q_i is updated with the old P_u, not the P_u we just changed.
        for f in range(n_factors):
            pu = P[u, f]
            qi = Q[i, f]
            P[u, f] += lr * (err * qi - reg * pu)
            Q[i, f] += lr * (err * pu - reg * qi)


class MatrixFactorization(Recommender):
    def __init__(self, n_factors: int = 50, lr: float = 0.005, reg: float = 0.02,
                 n_epochs: int = 20, seed: int = 42, like_threshold: float = 4.0):
        """
        n_factors: number of latent factors
        lr: SGD learning rate
        reg: L2 regularization strength (lambda)
        n_epochs: number of training passes over the data
        seed: random seed for initialization and shuffling
        like_threshold: minimum rating for a book to count as "liked" in recommend() (same rule as ItemKNN)
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.like_threshold = like_threshold

    def fit(self, train: pd.DataFrame) -> "MatrixFactorization":
        # np.random.default_rng(seed): seeded random generator, so the same seed always gives the same result
        rng = np.random.default_rng(self.seed)

        if "user_id" not in train.columns or "book_id" not in train.columns or "rating" not in train.columns:
            raise ValueError("train must contain user_id, book_id, and rating columns")

        self.user_ids_ = np.sort(train["user_id"].unique())
        self.item_ids_ = np.sort(train["book_id"].unique())
        self.user_to_idx_ = {user_id: idx for idx, user_id in enumerate(self.user_ids_)}
        self.item_to_idx_ = {book_id: idx for idx, book_id in enumerate(self.item_ids_)}
        self.n_users = len(self.user_ids_)
        self.n_items = len(self.item_ids_)

        self.global_mean_ = float(train["rating"].mean())
        self.user_mean_ = train.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean_ = train.groupby("book_id")["rating"].mean().to_dict()
        self.seen_ = train.groupby("user_id")["book_id"].apply(set).to_dict()

        item_counts = train.groupby("book_id").size()
        self.item_popularity_ = item_counts.sort_values(ascending=False).index.to_numpy()

        # Biases start at 0; latent vectors start from small random values, normal(mean 0, std 0.1)
        self.user_bias_ = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias_ = np.zeros(self.n_items, dtype=np.float64)
        self.P_ = rng.normal(0.0, 0.1, size=(self.n_users, self.n_factors)).astype(np.float64)
        self.Q_ = rng.normal(0.0, 0.1, size=(self.n_items, self.n_factors)).astype(np.float64)

        rows = train["user_id"].map(self.user_to_idx_).to_numpy(dtype=np.int64)
        cols = train["book_id"].map(self.item_to_idx_).to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)

        for _ in range(self.n_epochs):
            # The shuffle order is made with the Python-side rng, as before.
            # Shuffling inside numba would use a different random generator and change results for the same seed.
            idx = rng.permutation(len(rows))
            # Only the inner loop that updates one rating at a time goes to the numba function
            # (it modifies the arrays in place, so nothing is returned)
            _sgd_epoch(idx, rows, cols, ratings, self.global_mean_,
                       self.user_bias_, self.item_bias_, self.P_, self.Q_,
                       float(self.lr), float(self.reg))

        # ── Preparation for recommend ──
        # Matrix positions of each user's "liked" books (rating ≥ like_threshold). Users with no liked book use all rated books.
        liked = train["rating"].to_numpy() >= self.like_threshold
        all_rated = pd.Series(cols).groupby(rows).apply(np.array).to_dict()
        self.liked_ = pd.Series(cols[liked]).groupby(rows[liked]).apply(np.array).to_dict()
        for uid, items in all_rated.items():
            self.liked_.setdefault(uid, items)

        # Normalize the book latent vectors to length 1. Dot product of normalized vectors = cosine similarity.
        # (Zero-length vectors are divided by 1 to avoid division by zero)
        norms = np.linalg.norm(self.Q_, axis=1, keepdims=True)
        self.Q_unit_ = self.Q_ / np.where(norms == 0, 1.0, norms)

        return self

    def predict(self, user_id: int, book_id: int) -> float:
        uid = self.user_to_idx_.get(user_id)
        iid = self.item_to_idx_.get(book_id)

        if uid is None and iid is None:
            return float(self.global_mean_)
        if uid is None:
            return float(np.clip(self.item_mean_.get(book_id, self.global_mean_), 1.0, 5.0))
        if iid is None:
            return float(np.clip(self.user_mean_.get(user_id, self.global_mean_), 1.0, 5.0))

        pred = self.global_mean_ + self.user_bias_[uid] + self.item_bias_[iid] + self.P_[uid] @ self.Q_[iid]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if k <= 0:
            return []

        seen = self.seen_.get(user_id, set())
        if user_id not in self.user_to_idx_:
            ranking = [book_id for book_id in self.item_popularity_ if book_id not in seen]
            return ranking[:k]

        uid = self.user_to_idx_[user_id]

        # Ranking score: sum of cosine similarities between latent vectors (Q) and the liked books
        # (same idea as ItemKNN's "similarity sum" ranking)
        #
        #   score(i) = Σ_j cos(Q_i, Q_j)      (j = books the user liked, rating ≥ like_threshold)
        #
        # Why we do not rank by predicted rating (validation, 2,000 users, results/mf_ranking_comparison.csv):
        #   Ranking by predicted rating gives Precision@10 0.005, below popularity (0.039). The ranking is almost
        #   decided by the item bias, so "books few people read but those readers rate highly" (median 191 ratings)
        #   are recommended to everyone.
        #   A predicted rating answers "would they like it if they read it", while Precision asks "did they actually
        #   read and like it", so the two do not match.
        #   Raising reg (0.0004), reg on biases only (0.016), filtering out books with few ratings (≤0.013) and
        #   dropping the biases (0.013) helped little; this method (0.052) was the only one above popularity.
        #   predict() (RMSE) is not changed.
        #
        # Since Σ_j (Q̂_i · Q̂_j) = Q̂_i · (Σ_j Q̂_j), we first add up the liked books' vectors and then score
        # all 10,000 books with one matrix product.
        # (Q̂ = Q_unit_, normalized to length 1. Negative similarities are added as they are, as in the experiment)
        scores = self.Q_unit_ @ self.Q_unit_[self.liked_[uid]].sum(axis=0)

        # Books already read get -inf so they can never be in the top k.
        # seen_ is a set of book_ids, so convert them to matrix positions and set them at once
        # (they were in train, so they are always in the mapping).
        if seen:
            scores[[self.item_to_idx_[b] for b in seen]] = -np.inf

        candidate_count = self.n_items - len(seen)
        if candidate_count <= 0:
            return []

        top_k = min(k, candidate_count)
        # Instead of sorting all 10,000 scores, np.argpartition picks the top top_k first and only those are sorted.
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return self.item_ids_[top_idx].astype(int).tolist()
