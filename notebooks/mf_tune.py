"""MatrixFactorization hyperparameter tuning: compares n_factors (50, 100) x reg (0.02, 0.05) = 4 combinations on val.

Run: python notebooks/mf_tune.py   (about 2~4 minutes of training per combination)
Output: results/mf_tuning.csv (one row per combination, with both RMSE and Precision@10)
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
from src.models.mf import MatrixFactorization

# ── Search range ───────────────────────────────────────────────────
# lr (0.005), n_epochs (20) and like_threshold (4.0) stay at their defaults; only these two are changed.
#   n_factors: latent vector size. Larger captures taste in more detail but raises overfitting risk and training time.
#   reg      : L2 regularization strength. Larger keeps P, Q and biases small to prevent overfitting, but too large underfits.
N_FACTORS = [50, 100]
REGS = [0.02, 0.05]

# ── Evaluation settings ────────────────────────────────────────────
# Same conditions as knn_tune.py and mf_dev.py → directly comparable with the ItemKNN results.
# The seed is fixed, so every combination is evaluated on "the same 50,000 ratings and 2,000 users".
SAMPLE_N = 50_000
MAX_USERS = 2_000

# Tune on val only (test is used once, after choosing the final model).
train, val, _ = load_data()

rows = []
for n_factors, reg in product(N_FACTORS, REGS):
    t0 = time.time()
    model = MatrixFactorization(n_factors=n_factors, reg=reg).fit(train)
    fit_sec = time.time() - t0

    # RMSE/MAE evaluate predict (predicted rating); Precision/Recall evaluate recommend (Q similarity-sum ranking).
    # The two metrics look at different parts of the model, so a setting good for one is not guaranteed to be good for the other; both are recorded.
    t0 = time.time()
    res = {"n_factors": n_factors, "reg": reg}
    res.update(evaluate_rating(model, val, sample_n=SAMPLE_N))
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    res["fit_sec"] = round(fit_sec, 1)
    res["eval_sec"] = round(time.time() - t0, 1)
    rows.append(res)

    print(f"n_factors={n_factors:<4} reg={reg:<5} rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
    # Save after every combination → results so far are kept even if the run stops
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "mf_tuning.csv", index=False, float_format="%.4f")

# ── Summary ───────────────────────────────────────────────────────
print("\nAll results (results/mf_tuning.csv):")
print(out.to_string(index=False))
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]
print(f"\nLowest RMSE      : n_factors={best_rmse.n_factors:.0f}, reg={best_rmse.reg} → {best_rmse.rmse:.4f}")
print(f"Best Precision@10: n_factors={best_prec.n_factors:.0f}, reg={best_prec.reg} → {best_prec['precision@10']:.4f}")
