"""MatrixFactorization 하이퍼파라미터 튜닝: n_factors(50, 100) x reg(0.02, 0.05) = 4가지 조합을 val로 비교한다.

실행: python notebooks/mf_tune.py   (조합마다 학습 약 2~4분)
결과: results/mf_tuning.csv (조합마다 한 줄, RMSE와 Precision@10을 함께 기록)
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
from src.models.mf import MatrixFactorization

# ── 탐색 범위 ──────────────────────────────────────────────────────
# lr(0.005), n_epochs(20), like_threshold(4.0)는 기본값으로 고정하고 이 두 가지만 바꾼다.
#   n_factors: 잠재벡터 차원. 크면 취향을 더 세밀하게 표현하지만 과적합 위험과 학습 시간이 늘어난다.
#   reg      : L2 정규화 강도. 크면 P, Q, bias 값을 작게 눌러 과적합을 막지만, 너무 크면 덜 맞춘다.
N_FACTORS = [50, 100]
REGS = [0.02, 0.05]

# ── 평가 설정 ──────────────────────────────────────────────────────
# knn_tune.py, mf_dev.py와 같은 조건 → ItemKNN 결과와 그대로 비교할 수 있다.
# seed가 고정이라 모든 조합이 "같은 5만 건, 같은 2,000명"으로 평가된다.
SAMPLE_N = 50_000
MAX_USERS = 2_000

# 튜닝은 val로만 한다 (test는 최종 모델을 고른 뒤 한 번만).
train, val, _ = load_data()

rows = []
for n_factors, reg in product(N_FACTORS, REGS):
    t0 = time.time()
    model = MatrixFactorization(n_factors=n_factors, reg=reg).fit(train)
    fit_sec = time.time() - t0

    # RMSE/MAE는 predict(예상 평점), Precision/Recall은 recommend(Q 유사도 합 순위)를 평가한다.
    # 두 지표가 모델의 다른 부분을 보기 때문에, 한쪽에 좋은 설정이 다른 쪽에도 좋다는 보장이 없어서 둘 다 기록한다.
    t0 = time.time()
    res = {"n_factors": n_factors, "reg": reg}
    res.update(evaluate_rating(model, val, sample_n=SAMPLE_N))
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    res["fit_sec"] = round(fit_sec, 1)
    res["eval_sec"] = round(time.time() - t0, 1)
    rows.append(res)

    print(f"n_factors={n_factors:<4} reg={reg:<5} rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
    # 한 조합 끝날 때마다 저장 → 중간에 멈춰도 그때까지의 결과는 남는다
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "mf_tuning.csv", index=False, float_format="%.4f")

# ── 요약 ──────────────────────────────────────────────────────────
print("\n전체 결과 (results/mf_tuning.csv):")
print(out.to_string(index=False))
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]
print(f"\nRMSE 최저       : n_factors={best_rmse.n_factors:.0f}, reg={best_rmse.reg} → {best_rmse.rmse:.4f}")
print(f"Precision@10 최고: n_factors={best_prec.n_factors:.0f}, reg={best_prec.reg} → {best_prec['precision@10']:.4f}")
