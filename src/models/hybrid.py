import pandas as pd
from collections import defaultdict
from .base import Recommender


class Hybrid(Recommender):
    """
    협업 필터링(CF)과 콘텐츠 기반(Content-based) 모델을 결합한 하이브리드 추천 모델.
    - predict: 두 모델의 예측 평점을 가중 평균하여 반환
    - recommend: 두 모델의 추천 순위를 역수 순위 가중합(Reciprocal Rank Fusion)으로 결합
    """

    def __init__(self, cf_model: Recommender, content_model: Recommender, alpha: float = 0.7, recommend_pool: int = 200):
        """
        Args:
            cf_model: 미리 튜닝된 협업 필터링 모델 객체 (예: ItemKNN)
            content_model: 미리 튜닝된 콘텐츠 기반 모델 객체 (예: ContentBased)
            alpha: CF 모델의 반영 비율 (0.0 ~ 1.0). Content 비율은 (1 - alpha)
            recommend_pool: 순위 결합을 위해 각 모델에서 1차로 추출할 후보 도서의 수
        """
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha
        self.recommend_pool = recommend_pool

    def fit(self, train: pd.DataFrame):
        """두 개별 모델을 각각 피팅합니다."""
        #self.cf_model.fit(train)
        #self.content_model.fit(train)
        return self

    def predict(self, user_id: int, book_id: int) -> float:
        """
        두 모델의 예측 평점을 지정된 비율(alpha)로 혼합하여 최종 평점을 계산합니다.
        """
        pred_cf = self.cf_model.predict(user_id, book_id)
        pred_content = self.content_model.predict(user_id, book_id)
        
        return (self.alpha * pred_cf) + ((1.0 - self.alpha) * pred_content)

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """
        상위 추천 목록을 가져와 순위 기반으로 가중합산 후 최종 Top-K를 반환합니다.
        (역수 순위 가중합: alpha / rank)
        """
        # 1. 각 모델에서 설정된 풀 크기만큼 추천 도서 추출
        cf_recs = self.cf_model.recommend(user_id, k=self.recommend_pool)
        content_recs = self.content_model.recommend(user_id, k=self.recommend_pool)
        
        scores = defaultdict(float)
        
        # 2. CF 모델 추천 도서에 순위 점수 부여 (1위부터 시작이므로 start=1)
        for rank, book_id in enumerate(cf_recs, start=1):
            scores[book_id] += self.alpha / rank
            
        # 3. Content 모델 추천 도서에 순위 점수 부여
        for rank, book_id in enumerate(content_recs, start=1):
            scores[book_id] += (1.0 - self.alpha) / rank
            
        # 4. 합산된 점수를 기준으로 내림차순 정렬 후 상위 k권 추출
        sorted_books = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in sorted_books][:k]