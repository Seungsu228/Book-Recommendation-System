import pandas as pd


class Recommender:
    """Every model inherits from this class and implements the three methods below."""

    def fit(self, train: pd.DataFrame) -> "Recommender":
        """Learns from the train ratings and returns self."""
        raise NotImplementedError

    def predict(self, user_id: int, book_id: int) -> float:
        """Predicted rating (1~5). Returns a value without errors even for unseen users/books."""
        raise NotImplementedError

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """Returns the top k book_ids, excluding books the user already rated in train."""
        raise NotImplementedError
