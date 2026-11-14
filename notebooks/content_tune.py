"""Content-based 모델 하이퍼파라미터 튜닝
실행: python notebooks/content_tune.py
결과: results/content_tuning.csv (조합마다 한 줄)
"""
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.preprocess import load_data,load_books
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.content import ContentBased

# ── 1. 하이퍼파라미터 탐색 범위 ────────────────────────
# 총 32가지 조합 (2 x 2 x 2 x 2 x 2)
MIN_DFS = [2, 5]
MAX_DFS = [0.7, 0.9]
W_TAGS = [1.0, 2.0]
W_AUTHORS = [0.5, 1.0]
W_TITLES = [0.3, 0.8]

# ── 2. 평가 설정 ───────────────────────────────────
# 타 모델과의 공정한 비교를 위해 평가 데이터 수와 사용자 수를 고정하여 샘플링
SAMPLE_N = 50_000
MAX_USERS = 2_000

print("데이터를 로드하는 중입니다...")
train, val, _ = load_data()
# Content-based 모델 특성상 도서 메타데이터(tags, authors, title) 필요
books = load_books()

rows = []
print("튜닝을 시작합니다...\n")

for min_df, max_df, w_tags, w_authors, w_title in product(MIN_DFS, MAX_DFS, W_TAGS, W_AUTHORS, W_TITLES):
    t0 = time.time()
    
    # 모델 초기화 및 학습
    model = ContentBased(
        books=books, 
        min_df=min_df, 
        max_df=max_df,
        w_tags=w_tags, 
        w_authors=w_authors, 
        w_title=w_title
    ).fit(train)
    
    fit_sec = time.time() - t0
    t0 = time.time()
    
    res = {
        "min_df": min_df, "max_df": max_df, 
        "w_tags": w_tags, "w_authors": w_authors, "w_title": w_title
    }
    
    # ── 3. 모델 평가 ────────────────────────────────
    res.update(evaluate_rating(model, val, sample_n=SAMPLE_N))
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    res["fit_sec"] = round(fit_sec, 1)
    res["eval_sec"] = round(time.time() - t0, 1)
    rows.append(res)

    # ── 4. 결과 출력 및 중간 저장 ────────────────────
    print(f"min={min_df:<2} max={max_df:<3} tags={w_tags:<3} auth={w_authors:<3} title={w_title:<3} | "
          f"rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
          
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    # 루프마다 덮어쓰기 저장 (중단 시 데이터 유실 방지)
    out.to_csv(ROOT / "results" / "content_tuning.csv", index=False, float_format="%.4f")

# ── 5. 최종 결과 요약 ──────────────────────────────
print("\n전체 결과 (results/content_tuning.csv):")
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]

print(f"\nRMSE 최저        : w_tags={best_rmse.w_tags}, w_authors={best_rmse.w_authors}, w_title={best_rmse.w_title} → {best_rmse.rmse:.4f}")
print(f"Precision@10 최고: w_tags={best_prec.w_tags}, w_authors={best_prec.w_authors}, w_title={best_prec.w_title} → {best_prec['precision@10']:.4f}")