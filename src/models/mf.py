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
                 n_epochs: int = 20, seed: int = 42, like_threshold: float = 4.0):
        """
        n_factors: latent factor 수
        lr: SGD learning rate
        reg: L2 regularization 강도 (lambda)
        n_epochs: 학습 반복 횟수
        seed: 초기화 랜덤 시드
        like_threshold: recommend()에서 "좋아한 책"으로 볼 최소 평점 (ItemKNN과 같은 기준)
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
        self.like_threshold = like_threshold

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

        # ── recommend용 준비 ──
        # 사용자별 "좋아한 책"(평점 ≥ like_threshold)의 행렬 위치. 좋아한 책이 없는 사용자는 평가한 책 전체를 쓴다.
        liked = train["rating"].to_numpy() >= self.like_threshold
        all_rated = pd.Series(cols).groupby(rows).apply(np.array).to_dict()
        self.liked_ = pd.Series(cols[liked]).groupby(rows[liked]).apply(np.array).to_dict()
        for uid, items in all_rated.items():
            self.liked_.setdefault(uid, items)

        # 도서 잠재벡터를 길이 1로 정규화해 둔다. 정규화된 벡터끼리의 내적 = cosine 유사도.
        # (길이 0인 벡터는 나눗셈 에러가 나지 않도록 1로 나눈다)
        norms = np.linalg.norm(self.Q_, axis=1, keepdims=True)
        self.Q_unit_ = self.Q_ / np.where(norms == 0, 1.0, norms)

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

        # 순위 점수: 좋아한 책들과의 잠재벡터(Q) cosine 유사도 합 (ItemKNN의 "유사도 합" 순위와 같은 원리)
        #
        #   score(i) = Σ_j cos(Q_i, Q_j)      (j = 사용자가 좋아한 책, 평점 ≥ like_threshold)
        #
        # 예상 평점 순을 쓰지 않는 이유 (val, 사용자 2,000명, results/mf_ranking_comparison.csv):
        #   예상 평점 순은 Precision@10 0.005로 인기순(0.039)보다 낮다. 순위가 item bias로 거의 정해져서
        #   "읽은 사람은 높게 주지만 읽는 사람이 적은 책"(평점 수 중앙값 191개)이 모두에게 추천된다.
        #   예상 평점은 "읽으면 좋아할까"이고, Precision은 "실제로 읽고 좋아했나"라서 둘이 어긋난다.
        #   reg 올리기(0.0004), bias에만 reg(0.016), 평점 적은 책 거르기(≤0.013), bias 빼기(0.013)는 효과가 작았고,
        #   이 방식이 0.052로 유일하게 인기순보다 높았다. predict(RMSE)는 바뀌지 않는다.
        #
        # Σ_j (Q̂_i · Q̂_j) = Q̂_i · (Σ_j Q̂_j) 이므로, 좋아한 책 벡터를 먼저 더한 뒤 행렬 곱 한 번으로 1만 권 점수를 구한다.
        # (Q̂ = 길이 1로 정규화한 Q_unit_. 음의 유사도도 그대로 더한다 — 실험에서 측정한 방식과 같게)
        scores = self.Q_unit_ @ self.Q_unit_[self.liked_[uid]].sum(axis=0)

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
