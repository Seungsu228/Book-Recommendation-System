import numpy as np
import pandas as pd
import scipy.sparse as sp

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

    def fit(self, train: pd.DataFrame) -> "ItemKNN":
        # ── 1단계: ID → 행렬 위치 매핑 ─────────────────────────────
        # user_id, book_id는 1, 5, 9832처럼 띄엄띄엄한 숫자라 그대로 행렬 위치로 쓸 수 없다.
        # 그래서 고유 ID를 정렬한 뒤 0, 1, 2, ... 순서의 위치를 붙여준다.
        self.user_ids_ = np.sort(train["user_id"].unique())   # 위치 → user_id
        self.item_ids_ = np.sort(train["book_id"].unique())   # 위치 → book_id
        self.uidx_ = {u: i for i, u in enumerate(self.user_ids_)}  # user_id → 위치
        self.iidx_ = {b: i for i, b in enumerate(self.item_ids_)}  # book_id → 위치
        n_users, n_items = len(self.user_ids_), len(self.item_ids_)

        # ── 2단계: 사용자 × 도서 평점 행렬 (sparse) ────────────────
        # 평점 한 줄(user_id, book_id, rating)을 (행 위치, 열 위치, 값)으로 바꾼다.
        rows = train["user_id"].map(self.uidx_).to_numpy()
        cols = train["book_id"].map(self.iidx_).to_numpy()
        vals = train["rating"].to_numpy(dtype=np.float32)

        # 53,000 x 10,000 행렬 중 실제로 값이 있는 칸은 약 1%뿐이라,
        # 값이 있는 칸만 저장하는 sparse 행렬(CSR)을 쓴다. 일반 배열이면 약 2GB가 필요하다.
        # R_[u, i] = 사용자 u가 도서 i에 준 평점 (안 읽었으면 0)
        self.R_ = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))

        # ── 3단계: 평균 구하기 ─────────────────────────────────────
        # predict에서 "도서 평균 + 이웃들이 알려주는 보정값"으로 예측하기 때문에 도서별 평균이 필요하다.
        # 사용자 평균, 전체 평균은 처음 보는 사용자/도서가 들어왔을 때 대신 돌려줄 값이다.
        self.global_mean_ = float(vals.mean())                     # 전체 평균 평점

        # np.bincount(위치, weights=평점) = 위치별 평점 합, np.bincount(위치) = 위치별 평점 개수
        # 합 / 개수 = 평균. 반복문 없이 한 번에 계산된다.
        item_cnt = np.bincount(cols, minlength=n_items)           # 도서별 평점 개수
        user_cnt = np.bincount(rows, minlength=n_users)           # 사용자별 평점 개수
        self.item_mean_ = (np.bincount(cols, weights=vals, minlength=n_items)
                           / np.maximum(item_cnt, 1)).astype(np.float32)   # 도서별 평균
        self.user_mean_ = (np.bincount(rows, weights=vals, minlength=n_users)
                           / np.maximum(user_cnt, 1)).astype(np.float32)   # 사용자별 평균

        # 평점 개수가 많은 순서 = 인기순. 처음 보는 사용자에게 추천할 때 쓴다.
        self.popular_ = np.argsort(-item_cnt)

        return self
