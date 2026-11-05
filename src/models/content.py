import pandas as pd

from .base import Recommender


class ContentBased(Recommender):
    def __init__(self, books: pd.DataFrame, min_df: int = 2, max_df: float = 0.8,
                 w_tags: float = 1.0, w_authors: float = 0.5, w_title: float = 0.3):
        """
        books: preprocess.load_books() 결과
        min_df, max_df: TF-IDF에서 너무 드물거나 흔한 단어 제거 기준
        w_tags, w_authors, w_title: 특성별 가중치
        """
        self.books = books
        self.min_df = min_df
        self.max_df = max_df
        self.w_tags = w_tags
        self.w_authors = w_authors
        self.w_title = w_title
