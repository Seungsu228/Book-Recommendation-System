"""ItemKNN hyperparameter tuning: compares 3 similarities x k 20/40 = 6 combinations on val.

Run: python notebooks/knn_tune.py
Output: results/knn_tuning.csv (one row per combination)
"""
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))  # add the repo root folder to the import path

import pandas as pd

from src.preprocess import load_data
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.knn import ItemKNN

# ── Search range ───────────────────────────────────────────────────
# min_support (5) and like_threshold (4.0) stay at their defaults; only these two are changed.
SIMILARITIES = ["cosine", "adjusted_cosine", "pearson"]
KS = [20, 40]

# ── Evaluation settings ────────────────────────────────────────────
# Same conditions as knn_dev.py, so the results can be compared with the numbers seen there (RMSE 0.860, P@10 0.141).
# - RMSE/MAE: 50,000 random val ratings (predict works one rating at a time, so the full val set takes long)
# - Precision/Recall@K: 2,000 random val users
# The seed is fixed, so all 6 combinations are evaluated on "the same 50,000 ratings and 2,000 users" → a fair comparison.
SAMPLE_N = 50_000
MAX_USERS = 2_000

# Tune on val only. test must be used only once, after choosing the final model,
# so the hyperparameters are not "chosen to fit test".
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

    # Print and save after every combination → results so far are kept even if the run stops
    print(f"{sim:>15} k={k:<3} rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "knn_tuning.csv", index=False, float_format="%.4f")

# ── Summary ───────────────────────────────────────────────────────
# RMSE (rating prediction) and Precision@10 (ranking) can have different winners, so both are shown.
print("\nAll results (results/knn_tuning.csv):")
print(out.to_string(index=False))
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]
print(f"\nLowest RMSE      : {best_rmse.similarity}, k={best_rmse.k} → {best_rmse.rmse:.4f}")
print(f"Best Precision@10: {best_prec.similarity}, k={best_prec.k} → {best_prec['precision@10']:.4f}")
