"""MF 순위 방식 진단: 예상 평점 순 추천이 왜 인기순보다 Precision@10이 낮은지 확인하고, 대안들을 비교한다.

실행: python notebooks/mf_ranking_diag.py   (MF를 5번 학습하므로 약 12분)
결과: results/mf_ranking_comparison.csv (대안마다 한 줄)

평가 조건은 knn_tune.py, mf_dev.py와 같다 (RMSE는 val 5만 건, Precision/Recall은 사용자 2,000명, seed 고정).
대안 이름 앞 글자(A~G)는 발표 자료의 표와 맞춘 것이다.
  A: 예상 평점 순 (원래 방식)          B: A + 평점 수 적은 책 거르기
  C: bias 빼고 P·Q만                   D: A + α·log(평점 수)  (인기도 가중)
  E: 좋아한 책들과의 Q cosine 유사도 합 (현재 mf.py의 recommend)
  F: reg를 올려 재학습 + A             G: bias에만 reg를 따로 줘서 재학습 + A
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # 레포 맨 위 폴더를 import 경로에 추가

import numpy as np
import pandas as pd
from numba import njit

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.baseline import Popularity
from src.models.mf import MatrixFactorization

SAMPLE_N = 50_000
MAX_USERS = 2_000

train, val, _ = load_data()
book_count = train.groupby("book_id").size()   # 도서별 train 평점 수 (= 인기도)
print("도서별 train 평점 수:", book_count.describe(percentiles=[.01, .1, .5]).round(1).to_dict(), flush=True)


# ── 순위 방식만 바꿔 끼우는 래퍼 ─────────────────────────────────────
# 학습된 MF는 그대로 두고 recommend의 "점수 계산"만 바꿔서 비교하기 위한 클래스.
# score_fn(model, uid) → 도서 1만 권의 점수 배열. 이미 읽은 책 제외, 상위 k개 고르기는 mf.py와 같다.
class Ranker:
    def __init__(self, model, score_fn):
        self.model, self.score_fn = model, score_fn

    def recommend(self, user_id, k=10):
        m = self.model
        if user_id not in m.user_to_idx_:            # 처음 보는 사용자는 원래 모델의 처리(인기순)를 따른다
            return m.recommend(user_id, k)
        scores = np.array(self.score_fn(m, m.user_to_idx_[user_id]), dtype=np.float64)
        scores[[m.item_to_idx_[b] for b in m.seen_[user_id]]] = -np.inf
        top = np.argpartition(-scores, k - 1)[:k]
        return m.item_ids_[top[np.argsort(-scores[top])]].tolist()


def predicted(m, uid):
    """A: 예상 평점 (predict와 같은 식, clip 없음). 순위에는 상수인 global_mean, b_u가 영향을 주지 않는다."""
    return m.global_mean_ + m.user_bias_[uid] + m.item_bias_ + m.Q_ @ m.P_[uid]


rows = []


def evaluate(model, name, rmse):
    """Precision/Recall@K와 함께, 추천된 책들이 얼마나 인기 있는 책인지도 기록한다."""
    res = {"method": name, "rmse": rmse}
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    # 사용자 300명 x 10권의 추천 도서가 train에서 평점 몇 개를 받았는지
    #   rec_count_median: 중앙값 (인기순이면 1만 개 이상, 마니아 책 위주면 수백 개)
    #   rec_count_lt100 : 평점 100개 미만 책의 비율
    users = np.random.default_rng(1).choice(val["user_id"].unique(), 300, replace=False)
    recs = np.concatenate([model.recommend(int(u), 10) for u in users])
    c = book_count.reindex(recs).to_numpy()
    res["rec_count_median"] = float(np.median(c))
    res["rec_count_lt100"] = float((c < 100).mean())
    rows.append(res)
    print(f"  {name:<40} rmse={rmse:.4f} p@10={res['precision@10']:.4f} "
          f"추천 도서 평점 수 중앙값={res['rec_count_median']:.0f}", flush=True)


def rmse_of(model):
    return evaluate_rating(model, val, sample_n=SAMPLE_N)["rmse"]


# ── 비교 기준: 인기순 ────────────────────────────────────────────────
pop = Popularity().fit(train)
evaluate(pop, "Popularity", rmse_of(pop))

# ── 기본 MF 학습 + 원인 진단 ─────────────────────────────────────────
t0 = time.time()
base = MatrixFactorization().fit(train)
print(f"기본 MF 학습 {time.time() - t0:.0f}초", flush=True)
base_rmse = rmse_of(base)   # 순위만 바꾸는 A~E는 같은 모델이라 RMSE가 모두 이 값
cnt = book_count.reindex(base.item_ids_).to_numpy()   # 행렬 위치 순서로 맞춘 평점 수

print("\n== 원인 진단 ==")
# (1) 가설 "평점이 적은 책의 item bias가 크게 튄다" 확인: bias와 평점 수의 상관
print("corr(item bias, log 평점 수):", round(np.corrcoef(base.item_bias_, np.log(cnt))[0, 1], 3))
for lo, hi in [(0, 100), (100, 300), (300, 1000), (1000, 10**6)]:
    mk = (cnt >= lo) & (cnt < hi)
    print(f"  평점 수 {lo}~{hi}: {mk.sum()}권, item bias 평균 {base.item_bias_[mk].mean():+.3f}, 표준편차 {base.item_bias_[mk].std():.3f}")

# (2) bias가 가장 큰 책들: 평점 수가 적어서 튄 것인지, 실제로 평균이 높은 책인지
titles = pd.read_csv(ROOT / "data" / "books.csv").set_index("book_id")["title"]
book_mean = train.groupby("book_id")["rating"].mean()
print("item bias 상위 5권 (평점 수, bias, 실제 평균):")
for i in np.argsort(-base.item_bias_)[:5]:
    b = base.item_ids_[i]
    print(f"  {cnt[i]:>5}개  {base.item_bias_[i]:+.3f}  {book_mean[b]:.2f}  {titles[b]}")

# (3) 예상 평점에서 "모두에게 같은 부분"(b_i)과 "개인화 부분"(P·Q) 중 어느 쪽이 순위를 좌우하는지
u0 = base.user_to_idx_[int(val["user_id"].iloc[0])]
print(f"도서 간 점수 차이(표준편차): item bias {base.item_bias_.std():.3f} vs P·Q {(base.Q_ @ base.P_[u0]).std():.3f}"
      "  → bias가 크면 사용자마다 비슷한 목록이 나온다")

# ── 순위 방식만 바꾸기 (재학습 없음, RMSE 동일) ──────────────────────────
print("\n== 순위 방식 비교 ==")
evaluate(Ranker(base, predicted), "A. predicted rating (current)", base_rmse)
for n in [100, 300, 1000]:
    # 평점 수가 n 미만인 책은 -inf로 후보에서 뺀다
    mask = np.where(cnt >= n, 0.0, -np.inf)
    evaluate(Ranker(base, lambda m, u, mask=mask: predicted(m, u) + mask),
             f"B. predicted, books with >= {n} ratings", base_rmse)
evaluate(Ranker(base, lambda m, u: m.Q_ @ m.P_[u]), "C. P.Q only (no bias)", base_rmse)
log_cnt = np.log(cnt)
for a in [0.1, 0.3, 1.0]:
    # 인기 있는 책일수록 점수를 더 준다. α가 커질수록 인기순에 가까워진다.
    evaluate(Ranker(base, lambda m, u, a=a: predicted(m, u) + a * log_cnt),
             f"D. predicted + {a}*log(count)", base_rmse)
evaluate(base, "E. sum cos(Q_i, Q_liked) (KNN-style)", base_rmse)   # mf.py의 recommend 그대로

# ── 재학습이 필요한 대안 ──────────────────────────────────────────────
print("\n== 재학습 대안 ==")
for reg in [0.05, 0.1]:
    # F: reg를 올리면 bias도 줄어들지 않을까? → P·Q도 같이 줄어서 오히려 bias 비중이 커진다
    m = MatrixFactorization(reg=reg).fit(train)
    evaluate(Ranker(m, predicted), f"F. reg={reg}, predicted", rmse_of(m))


# G: bias에만 reg를 따로(더 세게) 주는 SGD. mf.py의 _sgd_epoch와 같고, bias 갱신에 reg 대신 reg_bias를 쓰는 것만 다르다.
#    mf.py에는 넣지 않은 실험용 변형이라 여기에만 둔다.
@njit(cache=True)
def _sgd_epoch_bias_reg(order, rows, cols, ratings, global_mean,
                        user_bias, item_bias, P, Q, lr, reg, reg_bias):
    n_factors = P.shape[1]
    for pos in order:
        u = rows[pos]
        i = cols[pos]
        dot = 0.0
        for f in range(n_factors):
            dot += P[u, f] * Q[i, f]
        err = ratings[pos] - (global_mean + user_bias[u] + item_bias[i] + dot)
        user_bias[u] += lr * (err - reg_bias * user_bias[u])   # ← 여기와
        item_bias[i] += lr * (err - reg_bias * item_bias[i])   # ← 여기만 reg_bias
        for f in range(n_factors):
            pu = P[u, f]
            qi = Q[i, f]
            P[u, f] += lr * (err * qi - reg * pu)
            Q[i, f] += lr * (err * pu - reg * qi)


def fit_bias_reg(reg_bias):
    # n_epochs=0으로 fit하면 ID 매핑, 평균, 초기 P/Q만 만들고 학습은 하지 않는다.
    m = MatrixFactorization(n_epochs=0).fit(train)
    # mf.py의 fit과 같은 난수 순서를 만들기 위해: 같은 seed로 시작해 P, Q 초기화에 쓴 난수를 똑같이 한 번씩 뽑아 버린다.
    # 그래야 epoch마다 섞는 순서가 기본 MF와 같아져서, "reg_bias만 다른" 공정한 비교가 된다.
    rng = np.random.default_rng(m.seed)
    rng.normal(size=(m.n_users, m.n_factors))
    rng.normal(size=(m.n_items, m.n_factors))
    rows_ = train["user_id"].map(m.user_to_idx_).to_numpy(np.int64)
    cols_ = train["book_id"].map(m.item_to_idx_).to_numpy(np.int64)
    ratings = train["rating"].to_numpy(np.float64)
    for _ in range(20):
        _sgd_epoch_bias_reg(rng.permutation(len(rows_)), rows_, cols_, ratings, m.global_mean_,
                            m.user_bias_, m.item_bias_, m.P_, m.Q_, m.lr, m.reg, reg_bias)
    return m


for rb in [0.5, 2.0]:
    m = fit_bias_reg(rb)
    evaluate(Ranker(m, predicted), f"G. reg_bias={rb}, predicted", rmse_of(m))

# ── 저장 ──────────────────────────────────────────────────────────
out = pd.DataFrame(rows)[["method", "rmse", "precision@5", "recall@5", "precision@10",
                          "recall@10", "rec_count_median", "rec_count_lt100"]]
(ROOT / "results").mkdir(exist_ok=True)
out.to_csv(ROOT / "results" / "mf_ranking_comparison.csv", index=False, float_format="%.4f")
print("\n저장: results/mf_ranking_comparison.csv")
print(out.round(4).to_string(index=False))
