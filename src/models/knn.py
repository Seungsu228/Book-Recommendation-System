from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, k: int = 20, similarity: str = "cosine", min_support: int = 5):
        """
        k: 예측에 참고할 이웃 도서 수
        similarity: "cosine" | "adjusted_cosine" | "pearson"
        min_support: 유사도를 계산할 최소 공통 평가자 수
        """
        self.k = k
        self.similarity = similarity
        self.min_support = min_support
