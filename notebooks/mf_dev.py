"""MatrixFactorization 확인 스크립트: val 기준으로 BookMean, Popularity, ItemKNN과 비교한다.

실행: python notebooks/mf_dev.py
평가 조건은 notebooks/knn_tune.py와 같다 (RMSE는 val 5만 건, Precision/Recall은 사용자 2,000명, seed 고정).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # 레포 맨 위 폴더를 import 경로에 추가

import numpy as np
import pandas as pd

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.baseline import BookMean, Popularity
from src.models.knn import ItemKNN
from src.models.mf import MatrixFactorization

SAMPLE_N = 50_000   # RMSE/MAE를 계산할 val 평점 수
MAX_USERS = 2_000   # Precision/Recall@K를 계산할 val 사용자 수

train, val, _ = load_data()   # 비교는 val로만 한다 (test는 최종 모델을 고른 뒤 한 번만)
books = pd.read_csv(ROOT / "data" / "books.csv").set_index("book_id")["title"]
book_count = train.groupby("book_id").size()   # 도서별 train 평점 수 (= 인기도)

# ── 1. 학습 ──────────────────────────────────────────────────────
# ItemKNN은 knn_tune.py에서 고른 설정(pearson, k=40)을 쓴다.
# MF는 mf_tune.py에서 고른 설정(n_factors=100, reg=0.02)을 쓴다.
#   n_factors 50 → 100에서 RMSE는 0.002만 나빠지고 Precision@10은 0.052 → 0.072로 올랐다 (results/mf_tuning.csv).
#   mf.py의 기본값(n_factors=50)은 그대로 두고 여기서만 지정한다.
models = {
    "BookMean": BookMean(),
    "Popularity": Popularity(),
    "ItemKNN (pearson, k=40)": ItemKNN(k=40, similarity="pearson"),
    "MF": MatrixFactorization(n_factors=100, reg=0.02),
}
for name, m in models.items():
    t0 = time.time()
    m.fit(train)
    print(f"{name} fit: {time.time() - t0:.1f}초", flush=True)

# ── 2. MF 동작 확인 ───────────────────────────────────────────────
mf = models["MF"]
row = val.iloc[0]
print("\nval 첫 줄:", int(row.user_id), int(row.book_id), "실제", row.rating,
      "→ MF 예측", round(mf.predict(row.user_id, row.book_id), 3))
print("처음 보는 사용자/도서:", round(mf.predict(10**9, 2), 3),
      round(mf.predict(int(row.user_id), 10**9), 3))

u = int(row.user_id)
recs = mf.recommend(u, 5)
print(f"사용자 {u} 추천:", recs, "/ 이미 읽은 책과 겹침:", len(set(recs) & mf.seen_[u]))
for b in recs:
    # 추천된 책이 train에서 몇 명에게 평가받았는지 같이 본다 (인기 없는 책만 나오는지 확인용)
    print(f"    평점 {book_count[b]:>5}개  {books[b]}")

# ── 3. 평가 ──────────────────────────────────────────────────────
# 모든 모델을 같은 5만 건, 같은 2,000명으로 평가한다 (seed 고정).
rows = {}
for name, m in models.items():
    t0 = time.time()
    res = evaluate_rating(m, val, sample_n=SAMPLE_N)
    res.update(evaluate_topk(m, val, max_users=MAX_USERS))
    rows[name] = res
    print(f"{name} 평가: {time.time() - t0:.1f}초", flush=True)

result = pd.DataFrame(rows).T
print("\nval 결과")
print(result.round(4).to_string())

# ── 4. Precision@10: MF가 인기순보다 높은가? ────────────────────────
p_mf = result.loc["MF", "precision@10"]
p_pop = result.loc["Popularity", "precision@10"]
print(f"\nPrecision@10  MF {p_mf:.4f} vs Popularity {p_pop:.4f} →",
      "MF가 높음" if p_mf > p_pop else "MF가 낮음")

# 추천 목록이 얼마나 "인기 있는 책" 위주인지: 추천된 책들의 train 평점 수 중앙값
# (인기순은 수천 개, 덜 알려진 책 위주면 수십~수백 개로 나온다)
users = np.random.default_rng(1).choice(val["user_id"].unique(), 300, replace=False)
print("추천 도서의 train 평점 수 중앙값 (사용자 300명 x 10권):")
for name in ["Popularity", "ItemKNN (pearson, k=40)", "MF"]:
    recs = np.concatenate([models[name].recommend(int(u), 10) for u in users])
    print(f"  {name:<25} {np.median(book_count.reindex(recs)):.0f}")
