import numpy as np
import math
import scipy.sparse as sp
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
        tfidf_tags = tfidf.fit_transform(self.books['tags'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
        tfidf_authors = tfidf.fit_transform(self.books['authors'].fillna(''))
        tfidf_title = tfidf.fit_transform(self.books['title'].fillna(''))

        # (피드백 반영) 메모리 절약을 위해 가중치의 제곱근을 곱한 TF-IDF 행렬을 하나로 결합
        self.tfidf_combined = sp.hstack([
            math.sqrt(self.w_tags) * tfidf_tags,
            math.sqrt(self.w_authors) * tfidf_authors,
            math.sqrt(self.w_title) * tfidf_title
        ], format='csr')

        

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

        return self

    def predict(self, user_id: int, book_id: int) -> float:
        """가중 유사도를 기반으로 해당 도서에 대한 유저의 예상 평점을 1~5점 사이로 반환합니다."""

        # 신규 도서거나 신규 유저일 경우(콜드 스타트) 전체 평균 반환
        if book_id not in self.indices or user_id not in self.user_ratings:
            return self.global_mean
        
        target_idx = self.indices[book_id]
        user_history = self.user_ratings[user_id]

        # (피드백 반영)예측 시점에 타겟 책과 전체 책 데이터의 유사도만 계산
        target_sims = linear_kernel(self.tfidf_combined[target_idx], self.tfidf_combined)[0]

        sim_scores = []
        ratings = []


        for hist_book_id, rating in user_history.items():
            if hist_book_id in self.indices:
                hist_idx = self.indices[hist_book_id]
                #방금 계산한 한줄에서 가져오기
                sim = target_sims[hist_idx]  
        

        # 유사한 책을 읽은 기록이 없으면 전체 평균 반환
        if not sim_scores or sum(sim_scores) == 0:
            return self.global_mean

        # 유사도 점수를 가중치로 사용하여 평점의 가중 평균 계산
        weighted_rating = np.dot(sim_scores, ratings) / sum(sim_scores)

        # 최종 예측 점수가 1.0 ~ 5.0 사이를 벗어나지 않도록 클리핑
        return float(np.clip(weighted_rating, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        """특정 유저에게 예상 평점이 가장 높은 상위 K개의 도서를 추천합니다."""
        user_history = self.user_ratings.get(user_id, {})
        
        # 1. 유저가 이미 읽은책은 추천에서 제외하기 위해 안 읽은 책 목록 추출
        unread_books = [book_id for book_id in self.indices.keys() if book_id not in user_history]
        
        # 2. 안 읽은 책들에 대해 예상 평점계산
        predictions = []
        for book_id in unread_books:
            est_rating = self.predict(user_id, book_id)
            predictions.append((book_id, est_rating))
            
        # 3. 예상 평점 내림차순(높은 순)으로 정렬
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 상위 K개의 도서 ID만 리스트로 반환
        return [book_id for book_id, _ in predictions[:k]]
    