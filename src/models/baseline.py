from .base import Recommender


class GlobalMean(Recommender):
    """모든 예측을 전체 평균 평점으로 한다."""
    def __init__(self):
        pass


class BookMean(Recommender):
    """도서별 평균 평점으로 예측한다."""
    def __init__(self):
        pass


class Popularity(Recommender):
    """평점 수가 많은 도서 순으로 추천한다."""
    def __init__(self):
        pass
