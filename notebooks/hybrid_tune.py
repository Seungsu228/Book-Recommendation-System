import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocess import load_data, load_books
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.knn import ItemKNN
from src.models.content import ContentBased
from src.models.hybrid import Hybrid

def main():
    print("Loading data...")
    train, val, _ = load_data()
    books = load_books()

    # Hybrid.fit does not retrain the sub-models, so train them here first
    knn_model = ItemKNN(k=40, similarity="pearson").fit(train)
    content_model = ContentBased(books=books).fit(train)
    
    # 1. Assemble the model
    model = Hybrid(cf_model=knn_model, content_model=content_model, alpha=0.5)
    
    print("\n[Step 1] Fitting the base models (KNN, Content)...")
    t_fit_start = time.time()
    model.fit(train)
    print(f"Fitting done! ({time.time() - t_fit_start:.0f}s)\n")
    
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("[Step 2] Evaluating alpha values to find the best one...")
    results = []
    
    for alpha in alphas:
        print(f"  -> evaluating alpha={alpha:.1f}... ", end='', flush=True)
        
        model.alpha = alpha
        
        t_eval_start = time.time()
        rmse_res = evaluate_rating(model, val, sample_n=50_000) 
        p10_res = evaluate_topk(model, val, max_users=2_000)
        eval_time = time.time() - t_eval_start
        
        # If a dict is returned, safely take only the number (float) (bracket typo fixed)
        rmse = rmse_res.get('rmse', list(rmse_res.values())[0]) if isinstance(rmse_res, dict) else rmse_res
        p10 = p10_res.get('precision@10', p10_res.get('p@10', list(p10_res.values())[0])) if isinstance(p10_res, dict) else p10_res
        
        print(f"done! (rmse={rmse:.4f} / p@10={p10:.4f} / {eval_time:.0f}s)")
        results.append({'alpha': alpha, 'rmse': rmse, 'p@10': p10})

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/hybrid_tuning.csv", index=False)
    print("\nTuning done! Results saved to results/hybrid_tuning.csv.")

if __name__ == "__main__":
    main()
