import pandas as pd


class Recommender:
    """모든 모델은 이 클래스를 상속하고 아래 세 메서드를 구현한다."""

    def fit(self, train: pd.DataFrame) -> "Recommender":
        """train 평점으로 학습하고 self를 반환한다."""
        raise NotImplementedError

    def predict(self, user_id: int, book_id: int) -> float:
        """예상 평점(1~5). 처음 보는 사용자/도서도 에러 없이 값을 반환한다."""
        raise NotImplementedError

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """train에서 이미 평가한 도서를 제외하고 상위 k개 book_id를 반환한다."""
        raise NotImplementedError
