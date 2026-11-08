"""Baseline recommenders: global/book/user mean for rating prediction, popularity for Top-N."""
import pandas as pd

from .base import Recommender


class _Base(Recommender):
    """Shared state: means, books each user has seen, and a popularity ranking."""

    def fit(self, train: pd.DataFrame) -> "_Base":
        self.global_mean_ = float(train["rating"].mean())
        self.seen_ = train.groupby("user_id")["book_id"].apply(set).to_dict()
        stats = train.groupby("book_id")["rating"].agg(["mean", "count"])
        self.book_mean_ = stats["mean"].to_dict()
        self.book_count_ = stats["count"].to_dict()
        self.user_mean_ = train.groupby("user_id")["rating"].mean().to_dict()
        self.ranking_ = (stats.sort_values(["count", "mean"], ascending=False)
                         .index.tolist())
        return self

    def predict(self, user_id: int, book_id: int) -> float:
        return self.global_mean_

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        seen = self.seen_.get(user_id, set())
        out = []
        for b in self.ranking_:
            if b not in seen:
                out.append(b)
                if len(out) == k:
                    break
        return out


class GlobalMean(_Base):
    """Predict the global mean rating for every pair. Recommend by popularity."""


class BookMean(_Base):
    """Predict each book's mean rating (global mean for unseen books)."""

    def predict(self, user_id: int, book_id: int) -> float:
        return self.book_mean_.get(book_id, self.global_mean_)


class UserMean(_Base):
    """Predict each user's mean rating (global mean for unseen users)."""

    def predict(self, user_id: int, book_id: int) -> float:
        return self.user_mean_.get(user_id, self.global_mean_)


class Popularity(_Base):
    """Recommend the books with the most ratings. Predict each book's mean rating."""

    def predict(self, user_id: int, book_id: int) -> float:
        return self.book_mean_.get(book_id, self.global_mean_)


class TopRated(_Base):
    """Recommend books with the highest mean rating among those with >= min_count ratings."""

    def __init__(self, min_count: int = 50):
        self.min_count = min_count

    def fit(self, train: pd.DataFrame) -> "TopRated":
        super().fit(train)
        self.ranking_ = sorted(
            (b for b, c in self.book_count_.items() if c >= self.min_count),
            key=lambda b: self.book_mean_[b], reverse=True)
        return self

    def predict(self, user_id: int, book_id: int) -> float:
        return self.book_mean_.get(book_id, self.global_mean_)
