import numpy as np
import pandas as pd
from numba import njit

from .base import Recommender


# ── SGD 한 epoch (numba로 컴파일) ───────────────────────────────────
# 파이썬 for문은 한 번 돌 때마다 인터프리터가 타입 확인, 객체 생성 등을 해서 느리다.
# 평점 478만 건 x 20 epoch = 약 1억 번을 돌면 그 오버헤드만으로 30분 이상 걸린다.
# @njit은 이 함수를 처음 호출할 때 기계어로 컴파일해서, C로 짠 반복문과 비슷한 속도로 돌게 한다.
# (nopython 모드라 numpy 배열과 숫자만 다룰 수 있다. 그래서 self 대신 배열들을 인자로 받는다.)
# cache=True: 컴파일 결과를 __pycache__에 저장해서 다음 실행부터는 컴파일 시간(수 초)을 건너뛴다.
@njit(cache=True)
def _sgd_epoch(order, rows, cols, ratings, global_mean,
               user_bias, item_bias, P, Q, lr, reg):
    """order 순서대로 평점을 하나씩 보며 b_u, b_i, P_u, Q_i를 갱신한다 (배열을 그 자리에서 수정)."""
    n_factors = P.shape[1]
    for pos in order:
        u = rows[pos]
        i = cols[pos]

        # 예측 = 전체평균 + b_u + b_i + P_u·Q_i  (기존 코드의 self.P_[u] @ self.Q_[i]를 반복문으로 계산)
        dot = 0.0
        for f in range(n_factors):
            dot += P[u, f] * Q[i, f]
        err = ratings[pos] - (global_mean + user_bias[u] + item_bias[i] + dot)

        # 바이어스 갱신: b ← b + lr·(err − reg·b)
        user_bias[u] += lr * (err - reg * user_bias[u])
        item_bias[i] += lr * (err - reg * item_bias[i])

        # 잠재 벡터 갱신: P_u ← P_u + lr·(err·Q_i − reg·P_u), Q_i ← Q_i + lr·(err·P_u − reg·Q_i)
        # 기존 코드에서 p_u, q_i를 .copy()해 둔 것과 같은 이유로, 갱신 전 값(pu, qi)을 먼저 꺼내 둔다.
        # 그래야 Q_i를 갱신할 때 "이미 바뀐 P_u"가 아니라 "바뀌기 전 P_u"를 쓴다.
        for f in range(n_factors):
            pu = P[u, f]
            qi = Q[i, f]
            P[u, f] += lr * (err * qi - reg * pu)
            Q[i, f] += lr * (err * pu - reg * qi)


class MatrixFactorization(Recommender):
    def __init__(self, n_factors: int = 50, lr: float = 0.005, reg: float = 0.02,
                 n_epochs: int = 20, seed: int = 42):
        """
        n_factors: latent factor 수
        lr: SGD learning rate
        reg: L2 regularization 강도 (lambda)
        n_epochs: 학습 반복 횟수
        seed: 초기화 랜덤 시드
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, train: pd.DataFrame) -> "MatrixFactorization":
        rng = np.random.default_rng(self.seed)

        if "user_id" not in train.columns or "book_id" not in train.columns or "rating" not in train.columns:
            raise ValueError("train must contain user_id, book_id, and rating columns")

        self.user_ids_ = np.sort(train["user_id"].unique())
        self.item_ids_ = np.sort(train["book_id"].unique())
        self.user_to_idx_ = {user_id: idx for idx, user_id in enumerate(self.user_ids_)}
        self.item_to_idx_ = {book_id: idx for idx, book_id in enumerate(self.item_ids_)}
        self.n_users = len(self.user_ids_)
        self.n_items = len(self.item_ids_)

        self.global_mean_ = float(train["rating"].mean())
        self.user_mean_ = train.groupby("user_id")["rating"].mean().to_dict()
        self.item_mean_ = train.groupby("book_id")["rating"].mean().to_dict()
        self.seen_ = train.groupby("user_id")["book_id"].apply(set).to_dict()

        item_counts = train.groupby("book_id").size()
        self.item_popularity_ = item_counts.sort_values(ascending=False).index.to_numpy()

        self.user_bias_ = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias_ = np.zeros(self.n_items, dtype=np.float64)
        self.P_ = rng.normal(0.0, 0.1, size=(self.n_users, self.n_factors)).astype(np.float64)
        self.Q_ = rng.normal(0.0, 0.1, size=(self.n_items, self.n_factors)).astype(np.float64)

        rows = train["user_id"].map(self.user_to_idx_).to_numpy(dtype=np.int64)
        cols = train["book_id"].map(self.item_to_idx_).to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)

        for _ in range(self.n_epochs):
            # 섞는 순서는 기존과 똑같이 파이썬 쪽 rng로 만든다.
            # numba 안에서 섞으면 난수 생성기가 달라져 같은 seed여도 결과가 바뀌기 때문이다.
            idx = rng.permutation(len(rows))
            # 평점 하나씩 갱신하는 안쪽 반복문만 numba 함수로 넘긴다 (배열을 그 자리에서 수정하므로 반환값 없음)
            _sgd_epoch(idx, rows, cols, ratings, self.global_mean_,
                       self.user_bias_, self.item_bias_, self.P_, self.Q_,
                       float(self.lr), float(self.reg))

        return self

    def predict(self, user_id: int, book_id: int) -> float:
        uid = self.user_to_idx_.get(user_id)
        iid = self.item_to_idx_.get(book_id)

        if uid is None and iid is None:
            return float(self.global_mean_)
        if uid is None:
            return float(np.clip(self.item_mean_.get(book_id, self.global_mean_), 1.0, 5.0))
        if iid is None:
            return float(np.clip(self.user_mean_.get(user_id, self.global_mean_), 1.0, 5.0))

        pred = self.global_mean_ + self.user_bias_[uid] + self.item_bias_[iid] + self.P_[uid] @ self.Q_[iid]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: int, k: int = 10) -> list[int]:
        if k <= 0:
            return []

        seen = self.seen_.get(user_id, set())
        if user_id not in self.user_to_idx_:
            ranking = [book_id for book_id in self.item_popularity_ if book_id not in seen]
            return ranking[:k]

        uid = self.user_to_idx_[user_id]

        # 모든 도서의 예상 평점을 한 번에 계산한다 (predict와 같은 식, 순위 기준도 기존과 같은 예상 평점).
        #   Q_ @ P_[uid] → (도서 수 × n_factors) @ (n_factors,) = 도서마다 P_u·Q_i 값이 담긴 길이 1만 벡터
        #   item_bias_도 길이 1만 벡터라 그대로 더하면 되고, global_mean_과 user_bias_[uid]는 숫자 하나라 전체에 더해진다.
        # 파이썬 반복문으로 1만 번 내적하던 것을 numpy(BLAS) 행렬 곱 한 번으로 바꿔서 훨씬 빠르다.
        # (predict와 달리 1~5로 clip하지 않는다. clip하면 5점을 넘는 책들이 모두 5.0으로 동점이 되어 순서가 사라지기 때문.
        #  기존 반복문도 clip하지 않았으므로 순위는 그대로다.)
        scores = self.global_mean_ + self.user_bias_[uid] + self.item_bias_ + self.Q_ @ self.P_[uid]

        # 이미 읽은 책은 -inf로 두어 상위 k개에 절대 뽑히지 않게 한다.
        # seen_은 book_id 집합이라, 행렬 위치로 바꿔서 한 번에 지정한다 (train에 있던 책이라 매핑에 항상 있음).
        if seen:
            scores[[self.item_to_idx_[b] for b in seen]] = -np.inf

        candidate_count = self.n_items - len(seen)
        if candidate_count <= 0:
            return []

        top_k = min(k, candidate_count)
        # 1만 개 전체를 정렬하지 않고, argpartition으로 상위 top_k개만 골라낸 뒤 그 top_k개만 정렬한다.
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return self.item_ids_[top_idx].astype(int).tolist()
