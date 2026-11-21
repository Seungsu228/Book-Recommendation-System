import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, k: int = 20, similarity: str = "cosine", min_support: int = 5,
                 like_threshold: float = 4.0):
        """
        k: number of neighbor books used for each book
        similarity: "cosine" | "adjusted_cosine" | "pearson"
        min_support: minimum number of common raters needed to compute a similarity
        like_threshold: minimum rating for a book to count as "liked" in recommend()
        """
        self.k = k
        self.similarity = similarity
        self.min_support = min_support
        self.like_threshold = like_threshold

    def fit(self, train: pd.DataFrame) -> "ItemKNN":
        # ── Step 1: map IDs to matrix positions ─────────────────────
        # user_id and book_id are sparse numbers such as 1, 5, 9832, so they cannot be used as matrix positions.
        # We sort the unique IDs and give them positions 0, 1, 2, ...
        self.user_ids_ = np.sort(train["user_id"].unique())   # position → user_id
        self.item_ids_ = np.sort(train["book_id"].unique())   # position → book_id
        self.uidx_ = {u: i for i, u in enumerate(self.user_ids_)}  # user_id → position
        self.iidx_ = {b: i for i, b in enumerate(self.item_ids_)}  # book_id → position
        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        # ── Step 2: user × book rating matrix (sparse) ──────────────
        # Each rating row (user_id, book_id, rating) becomes (row position, column position, value).
        rows = train["user_id"].map(self.uidx_).to_numpy()
        cols = train["book_id"].map(self.iidx_).to_numpy()
        vals = train["rating"].to_numpy(dtype=np.float32)

        # Only about 1% of the 53,000 x 10,000 cells have a value, so we use a sparse matrix (CSR),
        # which stores only the filled cells. A dense array would need about 2 GB.
        # scipy.sparse.csr_matrix((values, (row_positions, col_positions)), shape=(n_rows, n_cols))
        # R_[u, i] = rating user u gave book i (0 if not read)
        self.R_ = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # ── Step 3: means ───────────────────────────────────────────
        # predict() starts from the book mean and adds a correction from the neighbors, so we need book means.
        # The user mean and global mean are fallback values for users/books not seen in training.
        self.global_mean_ = float(vals.mean())                     # global mean rating

        # np.bincount(positions, weights=ratings) = sum of ratings per position;
        # np.bincount(positions) = number of ratings per position.
        # sum / count = mean, computed for all positions at once without a loop.
        # minlength makes the result cover every position, even ones with no ratings.
        item_cnt = np.bincount(cols, minlength=n_items)           # number of ratings per book
        user_cnt = np.bincount(rows, minlength=n_users)           # number of ratings per user
        self.item_mean_ = (np.bincount(cols, weights=vals, minlength=n_items)
                           / np.maximum(item_cnt, 1)).astype(np.float32)   # mean per book
        self.user_mean_ = (np.bincount(rows, weights=vals, minlength=n_users)
                           / np.maximum(user_cnt, 1)).astype(np.float32)   # mean per user

        # Most-rated first = popularity order. Used for users not seen in training.
        self.popular_ = np.argsort(-item_cnt)
        self.pop_score_ = item_cnt / item_cnt.max()   # popularity score in 0~1 (tie-break in recommend)

        # ── Step 4: book-to-book similarity → keep only the k most similar books per book ──
        # Similarity of books i and j = cosine similarity between their "rating column vectors".
        # Books that the same people rated in a similar way get a high similarity.

        # (1) Build matrix X with ratings adjusted according to the similarity type.
        if self.similarity == "cosine":            # raw ratings
            x_vals = vals
        elif self.similarity == "adjusted_cosine": # subtract user mean → corrects for generous/harsh raters
            x_vals = vals - self.user_mean_[rows]
        elif self.similarity == "pearson":         # subtract book mean → corrects for books that are liked/disliked overall
            x_vals = vals - self.item_mean_[cols]
        else:
            raise ValueError(f"unknown similarity: {self.similarity}")
        X = sp.csr_matrix((x_vals.astype(np.float32), (rows, cols)), shape=(n_users, n_items))

        # For min_support: 1 if rated, else 0. Dot product of two columns of B = number of people who rated both books.
        B = sp.csr_matrix((np.ones_like(vals), (rows, cols)), shape=(n_users, n_items))

        # cosine similarity = (column i · column j) / (|column i| × |column j|)
        XT, BT = X.T.tocsr(), B.T.tocsr()          # book × user (transposed so a book can be taken as a row)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0))).ravel()  # vector length per book |column i|
        norms[norms == 0] = 1.0                    # avoid division by zero for zero-length vectors

        # (2) A full 10,000 x 10,000 similarity matrix takes about 400 MB even in float32,
        #     so we compute it 1,000 books at a time and keep only the top k books per book.
        #     ("model size" approach from Sarwar et al., 2001: store only k neighbors)
        block = 1000
        kk = min(self.k, n_items - 1)
        nb_rows, nb_cols, nb_vals = [], [], []
        for start in range(0, n_items, block):
            end = min(start + block, n_items)

            sim = (XT[start:end] @ X).toarray()             # dot products, shape (block, all books)
            sim /= norms[start:end, None] * norms[None, :]  # divide by lengths → cosine similarity

            if self.min_support > 1:                        # too few common raters is unreliable → 0
                co = (BT[start:end] @ B).toarray()
                sim[co < self.min_support] = 0.0

            sim[np.arange(end - start), np.arange(start, end)] = 0.0  # a book is not its own neighbor
            sim[sim < 0] = 0.0                              # negative similarity (opposite taste) is not used

            # Positions of the top kk similarities in each row.
            # np.argpartition(a, kk, axis=1) moves the kk smallest values of -sim (= largest sim) to the
            # front of each row without fully sorting, which is much faster than np.argsort.
            top = np.argpartition(-sim, kk, axis=1)[:, :kk]
            # np.take_along_axis(sim, top, axis=1) picks, row by row, the similarity values at those positions.
            top_sim = np.take_along_axis(sim, top, axis=1)
            keep = top_sim > 0                              # do not store zero similarities

            nb_rows.append(np.repeat(np.arange(start, end), kk)[keep.ravel()])  # position of the book that owns the neighbor
            nb_cols.append(top[keep])                                         # position of the neighbor book
            nb_vals.append(top_sim[keep])                                     # similarity

        # (3) S_[i, j] = similarity between book i and its neighbor j. Sparse matrix with at most k values per book.
        self.S_ = sp.csr_matrix(
            (np.concatenate(nb_vals), (np.concatenate(nb_rows), np.concatenate(nb_cols))),
            shape=(n_items, n_items), dtype=np.float32)

        return self

    # ── Step 5: predict ──────────────────────────────────────────────
    # Prediction formula (same form as KNNWithMeans in the Surprise library, item-based):
    #
    #   r̂(u, i) = μ_i + Σ s(i, j)·(r_uj − μ_j) / Σ s(i, j)     (j = neighbors of book i that user u has read)
    #
    #   μ_i: mean rating of book i, s(i, j): similarity stored in Step 4, r_uj: rating user u gave book j
    #
    # Start from "this book's average", then correct it by how much higher/lower than average the user
    # rated the similar books, weighted by similarity. E.g. if the user rated similar books 1 point above
    # their averages, this book is predicted at its average + 1.

    def _user_deviation(self, uid: int):
        """Positions of the books user u rated, and each rating minus the book mean (r_uj − μ_j)"""
        row = self.R_[uid]                       # row of user u (only rated books have values)
        return row.indices, row.data - self.item_mean_[row.indices]

    def predict(self, user_id: int, book_id: int) -> float:
        iid = self.iidx_.get(book_id)            # None if the book was not in train
        uid = self.uidx_.get(user_id)            # None if the user was not in train

        # Unseen books/users have no neighbors to use, so return a mean (the interface promises no errors)
        if iid is None:
            return float(self.user_mean_[uid]) if uid is not None else self.global_mean_
        if uid is None:
            return float(self.item_mean_[iid])

        nb = self.S_[iid]                        # neighbors of book i: nb.indices = positions, nb.data = similarities
        rated, dev = self._user_deviation(uid)   # positions of books the user read, and deviations

        # Keep only neighbors the user has read.
        # np.intersect1d(a, b, return_indices=True) returns the common elements and their positions in a and in b.
        common, nb_pos, rated_pos = np.intersect1d(nb.indices, rated, return_indices=True)
        if len(common) == 0:                     # no overlap → no evidence, so use the book mean
            return float(self.item_mean_[iid])

        s = nb.data[nb_pos]                      # similarities of the overlapping neighbors
        pred = self.item_mean_[iid] + np.dot(s, dev[rated_pos]) / s.sum()   # formula above
        return float(np.clip(pred, 1.0, 5.0))   # keep the prediction in the rating range (1~5)

    # ── Step 6: recommend ────────────────────────────────────────────
    # Books are ranked not by the predicted rating from predict(), but by this score
    # (Deshpande & Karypis, 2004, "Item-based top-N recommendation algorithms"):
    #
    #   score(i) = Σ s(i, j)      (j = books the user liked, i.e. rating ≥ like_threshold)
    #
    # If we sort by predicted rating, a book that overlaps with only one of the user's books can get 5 stars
    # just because the user rated that one book highly, and it reaches the top on weak evidence.
    # The similarity sum grows when a book is similar to many liked books, so well-supported books rise.

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        uid = self.uidx_.get(user_id)
        if uid is None:                           # unseen users get the popularity order
            return self.item_ids_[self.popular_[:k]].tolist()

        row = self.R_[uid]
        rated, ratings = row.indices, row.data    # positions and ratings of the books the user rated
        liked = rated[ratings >= self.like_threshold]
        if len(liked) == 0:                       # no liked books → use all rated books instead
            liked = rated

        n_items = len(self.item_ids_)
        v = np.zeros(n_items, dtype=np.float32)
        v[liked] = 1.0                            # vector with 1 only at liked books

        # S_ @ v → for each book i, Σ_{j: liked} s(i, j). Scores for all 10,000 books in one matrix product.
        scores = (self.S_ @ v).astype(np.float64)
        scores += 1e-6 * self.pop_score_          # tiny bonus so that ties (especially 0) go to popular books

        scores[rated] = -np.inf                   # never recommend books already read
        k = min(k, n_items - len(rated))
        top = np.argpartition(-scores, k)[:k]     # positions of the top k (still unordered)
        top = top[np.argsort(-scores[top])]       # sort only those k by score
        return self.item_ids_[top].tolist()       # position → book_id
