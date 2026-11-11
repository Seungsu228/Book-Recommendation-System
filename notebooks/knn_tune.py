"""ItemKNN 하이퍼파라미터 튜닝: similarity 3종 x k 20/40 = 6가지 조합을 val로 비교한다.

실행: python notebooks/knn_tune.py
결과: results/knn_tuning.csv (조합마다 한 줄)
"""
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # 레포 맨 위 폴더를 import 경로에 추가

import pandas as pd

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.knn import ItemKNN

# ── 탐색 범위 ──────────────────────────────────────────────────────
# min_support(5), like_threshold(4.0)는 기본값으로 고정하고 이 두 가지만 바꾼다.
SIMILARITIES = ["cosine", "adjusted_cosine", "pearson"]
KS = [20, 40]

# ── 평가 설정 ──────────────────────────────────────────────────────
# knn_dev.py와 같은 조건으로 맞춰야 앞서 본 수치(RMSE 0.860, P@10 0.141)와 비교할 수 있다.
# - RMSE/MAE: val에서 무작위 5만 건 (predict가 한 건씩 계산이라 전체 val은 오래 걸림)
# - Precision/Recall@K: val 사용자 중 무작위 2,000명
# seed를 고정했으므로 6가지 조합 모두 "같은 5만 건, 같은 2,000명"으로 평가된다 → 공정한 비교.
SAMPLE_N = 50_000
MAX_USERS = 2_000

# 튜닝은 val로만 한다. test는 최종 모델을 고른 뒤 한 번만 써야
# "test에 맞춰 고른 하이퍼파라미터"가 되지 않는다.
train, val, _ = load_data()

rows = []
for sim, k in product(SIMILARITIES, KS):
    t0 = time.time()
    model = ItemKNN(k=k, similarity=sim).fit(train)
    fit_sec = time.time() - t0

    t0 = time.time()
    res = {"similarity": sim, "k": k}
    res.update(evaluate_rating(model, val, sample_n=SAMPLE_N))
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    res["fit_sec"] = round(fit_sec, 1)
    res["eval_sec"] = round(time.time() - t0, 1)
    rows.append(res)

    # 한 조합 끝날 때마다 출력 + 저장 → 중간에 멈춰도 그때까지의 결과는 남는다
    print(f"{sim:>15} k={k:<3} rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "knn_tuning.csv", index=False, float_format="%.4f")

# ── 요약 ──────────────────────────────────────────────────────────
# RMSE(평점 예측)와 Precision@10(추천 순위)은 서로 다른 조합이 1등일 수 있어서 따로 보여준다.
print("\n전체 결과 (results/knn_tuning.csv):")
print(out.to_string(index=False))
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]
print(f"\nRMSE 최저       : {best_rmse.similarity}, k={best_rmse.k} → {best_rmse.rmse:.4f}")
print(f"Precision@10 최고: {best_prec.similarity}, k={best_prec.k} → {best_prec['precision@10']:.4f}")
