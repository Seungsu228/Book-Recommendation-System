from .base import Recommender


class Hybrid(Recommender):
    """최종 점수 = alpha * CF 예측 + (1 - alpha) * Content 예측"""

    def __init__(self, cf_model: Recommender, content_model: Recommender, alpha: float = 0.5):
        """
        cf_model: ItemKNN 또는 MatrixFactorization
        content_model: ContentBased
        alpha: CF 점수 비중 (0~1)
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha
