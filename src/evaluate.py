import pandas as pd

from .models.base import Recommender


def evaluate_rating(model: Recommender, test: pd.DataFrame) -> dict:
    """{'rmse': ..., 'mae': ...}"""


def evaluate_topk(model: Recommender, test: pd.DataFrame, ks=(5, 10), threshold: float = 4.0) -> dict:
    """{'precision@5': ..., 'recall@5': ..., 'precision@10': ..., 'recall@10': ...}"""
