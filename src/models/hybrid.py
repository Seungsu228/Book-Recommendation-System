import pandas as pd
from collections import defaultdict
from .base import Recommender


class Hybrid(Recommender):
    """
    Hybrid recommender that combines a collaborative filtering (CF) model and a content-based model.
    - predict: returns a weighted average of the two models' predicted ratings
    - recommend: merges the two models' rankings with a weighted reciprocal-rank sum (Reciprocal Rank Fusion)
    """

    def __init__(self, cf_model: Recommender, content_model: Recommender, alpha: float = 0.9, recommend_pool: int = 200):
        """
        Args:
            cf_model: an already tuned collaborative filtering model (e.g. ItemKNN)
            content_model: an already tuned content-based model (e.g. ContentBased)
            alpha: weight of the CF model (0.0 ~ 1.0). The content weight is (1 - alpha)
            recommend_pool: number of candidate books taken from each model before the rankings are merged
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha
        self.recommend_pool = recommend_pool

    def fit(self, train: pd.DataFrame):
        """Uses the trained sub-models as they are. (Fit the sub-models before passing them in.)"""
        return self

    def predict(self, user_id: int, book_id: int) -> float:
        """
        Mixes the two models' predicted ratings with the given ratio (alpha) to get the final rating.
        """
        pred_cf = self.cf_model.predict(user_id, book_id)
        pred_content = self.content_model.predict(user_id, book_id)

        return (self.alpha * pred_cf) + ((1.0 - self.alpha) * pred_content)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """
        Takes each model's top list, adds up rank-based weighted scores, and returns the final top k.
        (weighted reciprocal rank: alpha / rank)
        """
        # 1. Get recommend_pool books from each model
        cf_recs = self.cf_model.recommend(user_id, k=self.recommend_pool)
        content_recs = self.content_model.recommend(user_id, k=self.recommend_pool)

        scores = defaultdict(float)

        # 2. Rank score for the CF model's books (ranks start at 1, so start=1)
        for rank, book_id in enumerate(cf_recs, start=1):
            scores[book_id] += self.alpha / rank

        # 3. Rank score for the content model's books
        for rank, book_id in enumerate(content_recs, start=1):
            scores[book_id] += (1.0 - self.alpha) / rank

        # 4. Sort by total score (descending) and take the top k
        sorted_books = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in sorted_books][:k]
