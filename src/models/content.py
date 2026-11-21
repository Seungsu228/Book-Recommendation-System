import numpy as np
import math
import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .base import Recommender

class ContentBased(Recommender):
    def __init__(self, books: pd.DataFrame, min_df: int = 2, max_df: float = 0.8,
                 w_tags: float = 1.0, w_authors: float = 1.0, w_title: float = 0.8):
        """
        books: output of preprocess.load_books()
        min_df, max_df: TF-IDF thresholds for dropping words that are too rare or too common
        w_tags, w_authors, w_title: weight of each feature

        Reference:
        - Weighted sum of per-feature TF-IDF similarities, and rating prediction on a 1~5 scale
        """
        self.books = books.reset_index(drop=True)
        self.min_df = min_df
        self.max_df = max_df
        self.w_tags = w_tags
        self.w_authors = w_authors
        self.w_title = w_title

        # Replace missing values with empty strings
        self.books['tags'] = self.books['tags'].fillna('')
        self.books['authors'] = self.books['authors'].fillna('')
        self.books['title'] = self.books['title'].fillna('')

        # TF-IDF vectorizer (sklearn): turns text into vectors where a word gets a high weight if it is
        # frequent in this book but rare across all books. Each row is normalized to length 1 (L2).
        # stop_words='english': drop common English filler words (the, of, and, ...)
        # min_df: drop words that appear in fewer than min_df books
        # max_df: drop words that appear in more than this fraction of books
        tfidf = TfidfVectorizer(stop_words='english', min_df=self.min_df, max_df=self.max_df)

        # Build a separate TF-IDF matrix for each feature
        tfidf_tags = tfidf.fit_transform(self.books['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
        tfidf_authors = tfidf.fit_transform(self.books['authors'].fillna(''))
        tfidf_title = tfidf.fit_transform(self.books['title'].fillna(''))

        # (from review feedback) To save memory, multiply each TF-IDF matrix by the square root of its weight
        # and join them into one matrix with scipy.sparse.hstack (side by side, kept sparse in CSR format).
        # The dot product of two combined vectors is then w_tags·cos_tags + w_authors·cos_authors + w_title·cos_title.
        self.tfidf_combined = sp.hstack([
            math.sqrt(self.w_tags) * tfidf_tags,
            math.sqrt(self.w_authors) * tfidf_authors,
            math.sqrt(self.w_title) * tfidf_title
        ], format='csr')



        # Mapping to find a book's DataFrame index from its book_id quickly
        self.indices = pd.Series(self.books.index, index=self.books['book_id']).drop_duplicates()

        # User rating history and global mean rating, filled in during fit
        self.user_ratings = {}
        self.global_mean = 3.0

    def fit(self, train: pd.DataFrame):
        """Takes the training data (train) and stores each user's rating history."""
        self.global_mean = train['rating'].mean()

        # Convert to a dictionary {user_id: {book_id: rating, ...}} to make predict fast
        self.user_ratings = train.groupby('user_id').apply(
            lambda x: dict(zip(x['book_id'], x['rating']))
        ).to_dict()

        return self

    def predict(self, user_id: int, book_id: int) -> float:
        """Returns the user's predicted rating for the book (1~5), based on weighted similarity."""

        # New book or new user (cold start): return the global mean
        if book_id not in self.indices or user_id not in self.user_ratings:
            return self.global_mean

        target_idx = self.indices[book_id]
        user_history = self.user_ratings[user_id]

        # (from review feedback) At prediction time, compute only the similarities between the target book and all books.
        # linear_kernel(X, Y) (sklearn) = dot products between rows of X and rows of Y.
        # The vectors are already normalized, so this gives the (weighted) cosine similarity without recomputing norms.
        target_sims = linear_kernel(self.tfidf_combined[target_idx], self.tfidf_combined)[0]

        sim_scores = []
        ratings = []


        for hist_book_id, rating in user_history.items():
            if hist_book_id in self.indices:
                hist_idx = self.indices[hist_book_id]
                # Take the value from the row we just computed
                sim = target_sims[hist_idx]
                if sim > 0:
                    sim_scores.append(sim)
                    ratings.append(rating)

        # No similar books in the history: return the global mean
        if not sim_scores or sum(sim_scores) == 0:
            return self.global_mean

        # Weighted average of the ratings, using the similarities as weights
        weighted_rating = np.dot(sim_scores, ratings) / sum(sim_scores)

        # Clip the final prediction so it stays between 1.0 and 5.0
        return float(np.clip(weighted_rating, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Scores all books at once by their similarity to the books the user read, and returns the top k."""
        user_history = self.user_ratings.get(user_id, {})
        hist = [(self.indices[b], r) for b, r in user_history.items() if b in self.indices]
        if not hist:
            return self.books["book_id"].head(k).tolist()

        hist_idx = np.array([i for i, _ in hist])
        hist_ratings = np.array([r for _, r in hist], dtype=float)

        # Similarities of books read (rows) x all books (columns) in one matrix product
        sims = linear_kernel(self.tfidf_combined[hist_idx], self.tfidf_combined)

        # Ranking score = sum of similarities to the liked books (rated 4 or more)
        # Ranking by weighted-average rating would put books with very low similarity at 5 stars at the top,
        # so we use the sum, which keeps the size of the similarity (same approach as recommend in knn.py)
        liked = hist_ratings >= 4
        if not liked.any():          # no liked books → use all books read
            liked[:] = True
        scores = sims[liked].sum(axis=0)

        scores[hist_idx] = -np.inf  # exclude books already read
        # np.argpartition: pick the top k without sorting everything, then sort only those k
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return self.books["book_id"].iloc[top].tolist()
