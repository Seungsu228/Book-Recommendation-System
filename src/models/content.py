import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from .base import Recommender

class ContentBased(Recommender):
    def __init__(self, books: pd.DataFrame, min_df: int = 2, max_df: float = 0.8,
                 w_tags: float = 1.0, w_authors: float = 0.5, w_title: float = 0.3):
        """
        books: preprocess.load_books() 결과
        min_df, max_df: TF-IDF에서 너무 드물거나 흔한 단어 제거 기준
        w_tags, w_authors, w_title: 특성별 가중치
        
        Reference:
        - TF-IDF 특성별 가중합  유사도 및 1~5점 평점 예측 
        """
        self.books = books.reset_index(drop=True)
        self.min_df = min_df
        self.max_df = max_df
        self.w_tags = w_tags
        self.w_authors = w_authors
        self.w_title = w_title

        # 결측치를 빈 문자열로 처리 
        self.books['tags'] = self.books['tags'].fillna('')
        self.books['authors'] = self.books['authors'].fillna('')
        self.books['title'] = self.books['title'].fillna('')

        # TF-IDF 벡터화기 초기화
        tfidf = TfidfVectorizer(stop_words='english', min_df=self.min_df, max_df=self.max_df)

        # 특성별로 각각 TF-IDF 행렬 생성
        tfidf_tags = tfidf.fit_transform(self.books['tags'])
        tfidf_authors = tfidf.fit_transform(self.books['authors'])
        tfidf_title = tfidf.fit_transform(self.books['title'])

        # 코사인 유사도 각각 계산
        sim_tags = linear_kernel(tfidf_tags, tfidf_tags)
        sim_authors = linear_kernel(tfidf_authors, tfidf_authors)
        sim_title = linear_kernel(tfidf_title, tfidf_title)

        # 팀에서 설정한 가중치를 반영하여 최종 유사도 행렬 계산
        self.cosine_sim = (self.w_tags * sim_tags) + \
                          (self.w_authors * sim_authors) + \
                          (self.w_title * sim_title)

        # book_id로 데이터프레임의 인덱스를 빠르게 찾기 위한 매핑
        self.indices = pd.Series(self.books.index, index=self.books['book_id']).drop_duplicates()

        # 학습 시 저장할 유저 평점 기록과 전체 평균 평점
        self.user_ratings = {}
        self.global_mean = 3.0

    def fit(self, train: pd.DataFrame):
        """학습 데이터(train)를 받아 유저별 평점 기록을 저장합니다."""
        self.global_mean = train['rating'].mean()
        
        # predict 연산 최적화를 위해 {user_id: {book_id: rating, ...}} 형태의 딕셔너리로 변환
        self.user_ratings = train.groupby('user_id').apply(
            lambda x: dict(zip(x['book_id'], x['rating']))
        ).to_dict()

    def predict(self, user_id: int, book_id: int) -> float:
        """가중 유사도를 기반으로 해당 도서에 대한 유저의 예상 평점을 1~5점 사이로 반환합니다."""
        # 신규 도서거나 신규 유저일 경우(콜드 스타트) 전체 평균 반환
        if book_id not in self.indices or user_id not in self.user_ratings:
            return self.global_mean

        target_idx = self.indices[book_id]
        user_history = self.user_ratings[user_id]

        sim_scores = []
        ratings = []

        # 유저가 과거에 읽은 책들과 타겟 책의 유사도를 모두 확인
        for hist_book_id, rating in user_history.items():
            if hist_book_id in self.indices:
                hist_idx = self.indices[hist_book_id]
                sim = self.cosine_sim[target_idx][hist_idx]

                if sim > 0:
                    sim_scores.append(sim)
                    ratings.append(rating)

        # 유사한 책을 읽은 기록이 없으면 전체 평균 반환
        if not sim_scores or sum(sim_scores) == 0:
            return self.global_mean

        # 유사도 점수를 가중치로 사용하여 평점의 가중 평균 계산
        weighted_rating = np.dot(sim_scores, ratings) / sum(sim_scores)

        # 최종 예측 점수가 1.0 ~ 5.0 사이를 벗어나지 않도록 클리핑
        return float(np.clip(weighted_rating, 1.0, 5.0))