"""Hyperparameter tuning for the content-based model
Run: python notebooks/content_tune.py
Output: results/content_tuning.csv (one row per combination)
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

# ── 1. Hyperparameter search range ────────────────────
# 32 combinations in total (2 x 2 x 2 x 2 x 2)
MIN_DFS = [2, 5]
MAX_DFS = [0.7, 0.9]
W_TAGS = [1.0, 2.0]
W_AUTHORS = [0.5, 1.0]
W_TITLES = [0.3, 0.8]

# ── 2. Evaluation settings ─────────────────────────
# The number of evaluation ratings and users is fixed for a fair comparison with the other models
SAMPLE_N = 50_000
MAX_USERS = 2_000

print("Loading data...")
train, val, _ = load_data()
# The content-based model needs book metadata (tags, authors, title)
books = load_books()

rows = []
print("Starting tuning...\n")

for min_df, max_df, w_tags, w_authors, w_title in product(MIN_DFS, MAX_DFS, W_TAGS, W_AUTHORS, W_TITLES):
    t0 = time.time()
    
    # Build and train the model
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
    
    # ── 3. Evaluate the model ──────────────────────
    res.update(evaluate_rating(model, val, sample_n=SAMPLE_N))
    res.update(evaluate_topk(model, val, max_users=MAX_USERS))
    res["fit_sec"] = round(fit_sec, 1)
    res["eval_sec"] = round(time.time() - t0, 1)
    rows.append(res)

    # ── 4. Print the result and save progress ──────
    print(f"min={min_df:<2} max={max_df:<3} tags={w_tags:<3} auth={w_authors:<3} title={w_title:<3} | "
          f"rmse={res['rmse']:.4f} p@10={res['precision@10']:.4f} "
          f"(fit {fit_sec:.0f}s, eval {res['eval_sec']:.0f}s)", flush=True)
          
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    # Overwrite the file every loop (so results are not lost if the run stops)
    out.to_csv(ROOT / "results" / "content_tuning.csv", index=False, float_format="%.4f")

# ── 5. Final summary ───────────────────────────
print("\nAll results (results/content_tuning.csv):")
best_rmse = out.loc[out["rmse"].idxmin()]
best_prec = out.loc[out["precision@10"].idxmax()]

print(f"\nLowest RMSE      : w_tags={best_rmse.w_tags}, w_authors={best_rmse.w_authors}, w_title={best_rmse.w_title} → {best_rmse.rmse:.4f}")
print(f"Best Precision@10: w_tags={best_prec.w_tags}, w_authors={best_prec.w_authors}, w_title={best_prec.w_title} → {best_prec['precision@10']:.4f}")