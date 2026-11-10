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

        # ── 4단계: 도서끼리 유사도 → 도서마다 가장 비슷한 k권만 저장 ──
        # 도서 i, j의 유사도 = 두 도서의 "평점 열 벡터" 사이의 코사인 유사도.
        # 같은 사람들이 비슷한 점수를 준 책일수록 유사도가 높다.

        # (1) 유사도 방식에 따라 평점을 보정한 행렬 X를 만든다.
        if self.similarity == "cosine":            # 평점 그대로
            x_vals = vals
        elif self.similarity == "adjusted_cosine": # 사용자 평균을 뺌 → 점수를 후하게/짜게 주는 습관 보정
            x_vals = vals - self.user_mean_[rows]
        elif self.similarity == "pearson":         # 도서 평균을 뺌 → 원래 평이 좋은/나쁜 책 보정
            x_vals = vals - self.item_mean_[cols]
        else:
            raise ValueError(f"unknown similarity: {self.similarity}")
        X = sp.csr_matrix((x_vals.astype(np.float32), (rows, cols)), shape=(n_users, n_items))

        # min_support 확인용: 평가했으면 1, 아니면 0인 행렬. B의 열끼리 내적 = 두 책을 모두 평가한 사람 수
        B = sp.csr_matrix((np.ones_like(vals), (rows, cols)), shape=(n_users, n_items))

        # 코사인 유사도 = (열 i · 열 j) / (|열 i| × |열 j|)
        XT, BT = X.T.tocsr(), B.T.tocsr()          # 도서 × 사용자 (도서를 행으로 꺼내기 쉽게 전치)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0))).ravel()  # 도서별 벡터 길이 |열 i|
        norms[norms == 0] = 1.0                    # 길이 0이면 나눗셈 에러가 나므로 1로 둔다

        # (2) 10,000 x 10,000 유사도 행렬을 한 번에 만들면 float32로도 약 400MB라,
        #     1,000권씩 나눠서 계산하고 도서마다 상위 k권만 남긴다.
        #     (Sarwar et al., 2001의 "model size" 방식: 이웃을 k개만 저장)
        block = 1000
        kk = min(self.k, n_items - 1)
        nb_rows, nb_cols, nb_vals = [], [], []
        for start in range(0, n_items, block):
            end = min(start + block, n_items)

            sim = (XT[start:end] @ X).toarray()             # (block, 전체 도서) 내적
            sim /= norms[start:end, None] * norms[None, :]  # 길이로 나눠 코사인 유사도로

            if self.min_support > 1:                        # 공통 평가자가 너무 적으면 믿을 수 없으니 0
                co = (BT[start:end] @ B).toarray()
                sim[co < self.min_support] = 0.0

            sim[np.arange(end - start), np.arange(start, end)] = 0.0  # 자기 자신은 이웃에서 제외
            sim[sim < 0] = 0.0                              # 음의 유사도(취향이 반대)는 사용하지 않음

            # 행마다 유사도 상위 kk개의 위치. argpartition은 전체 정렬보다 빠르게 상위 k개만 골라준다.
            top = np.argpartition(-sim, kk, axis=1)[:, :kk]
            top_sim = np.take_along_axis(sim, top, axis=1)
            keep = top_sim > 0                              # 유사도 0인 것은 저장하지 않음

            nb_rows.append(np.repeat(np.arange(start, end), kk)[keep.ravel()])  # 이웃의 주인 도서 위치
            nb_cols.append(top[keep])                                         # 이웃 도서 위치
            nb_vals.append(top_sim[keep])                                     # 유사도

        # (3) S_[i, j] = 도서 i의 이웃 j와의 유사도. 도서마다 최대 k개만 값이 있는 sparse 행렬.
        self.S_ = sp.csr_matrix(
            (np.concatenate(nb_vals), (np.concatenate(nb_rows), np.concatenate(nb_cols))),
            shape=(n_items, n_items), dtype=np.float32)

        return self

    # ── 5단계: predict ─────────────────────────────────────────────
    # 예측식 (Surprise 라이브러리의 KNNWithMeans와 같은 형태, 도서 기준):
    #
    #   r̂(u, i) = μ_i + Σ s(i, j)·(r_uj − μ_j) / Σ s(i, j)     (j = 도서 i의 이웃 중 사용자 u가 읽은 책)
    #
    #   μ_i: 도서 i의 평균 평점, s(i, j): 4단계에서 저장한 유사도, r_uj: 사용자 u가 책 j에 준 평점
    #
    # "이 책의 평균 점수"에서 출발해서, 비슷한 책들에 사용자가 평균보다 얼마나 높게/낮게 줬는지를
    # 유사도만큼 반영해 보정한다. 예) 비슷한 책들에 평균보다 1점씩 높게 줬다면 이 책도 평균 + 1점.

    def _user_deviation(self, uid: int):
        """사용자 u가 평가한 도서 위치들과, 각 평점에서 도서 평균을 뺀 값 (r_uj − μ_j)"""
        row = self.R_[uid]                       # 사용자 u의 행 (평가한 책만 값이 있음)
        return row.indices, row.data - self.item_mean_[row.indices]

    def predict(self, user_id: int, book_id: int) -> float:
        iid = self.iidx_.get(book_id)            # 없으면 None (train에 없던 도서)
        uid = self.uidx_.get(user_id)            # 없으면 None (train에 없던 사용자)

        # 처음 보는 도서/사용자는 이웃을 쓸 수 없으니 평균으로 대신한다 (에러 없이 값을 돌려주기로 한 약속)
        if iid is None:
            return float(self.user_mean_[uid]) if uid is not None else self.global_mean_
        if uid is None:
            return float(self.item_mean_[iid])

        nb = self.S_[iid]                        # 도서 i의 이웃들: nb.indices = 이웃 위치, nb.data = 유사도
        rated, dev = self._user_deviation(uid)   # 사용자가 읽은 책 위치들과 편차

        # 이웃 중 사용자가 읽은 책만 남긴다. intersect1d는 공통 원소와, 각 배열에서의 위치를 함께 돌려준다.
        common, nb_pos, rated_pos = np.intersect1d(nb.indices, rated, return_indices=True)
        if len(common) == 0:                     # 겹치는 책이 없으면 근거가 없으니 도서 평균
            return float(self.item_mean_[iid])

        s = nb.data[nb_pos]                      # 겹치는 이웃들의 유사도
        pred = self.item_mean_[iid] + np.dot(s, dev[rated_pos]) / s.sum()   # 위 예측식
        return float(np.clip(pred, 1.0, 5.0))   # 평점 범위(1~5)를 벗어나지 않게 자르기
