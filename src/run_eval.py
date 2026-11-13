"""Fit every model on the shared split and save comparison tables and charts.

Usage (from the repo root):
    python3 -m src.run_eval --users 2000          # quick check on a user subset
    python3 -m src.run_eval --skip-cold           # full train, validation comparison
    python3 -m src.run_eval --split test          # final run, use only once
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import evaluate_model
from .models.baseline import BookMean, GlobalMean, Popularity
from .preprocess import load_books, load_cold_start_split, load_data

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def model_factories(books, knn_k=40, knn_similarity="pearson"):
    """Return {name: factory}. Imports are lazy so one broken model does not stop the run."""
    def knn():
        from .models.knn import ItemKNN
        return ItemKNN(k=knn_k, similarity=knn_similarity)

    def mf():
        from .models.mf import MatrixFactorization
        return MatrixFactorization()

    def content():
        from .models.content import ContentBased
        return ContentBased(books)

    return {
        "GlobalMean": GlobalMean,
        "BookMean": BookMean,
        "Popularity": Popularity,
        "ItemKNN": knn,
        "MatrixFactorization": mf,
        "ContentBased": content,
    }


def subsample_users(n_users, seed, *frames):
    """Keep only the rows of n_users randomly chosen users in every DataFrame."""
    users = np.sort(frames[0]["user_id"].unique())
    keep = set(np.random.default_rng(seed).choice(users, size=min(n_users, len(users)), replace=False))
    return [f[f["user_id"].isin(keep)].reset_index(drop=True) for f in frames]


def run(factories, train, test, books, eval_kwargs, hybrid_alpha):
    """Fit and evaluate each model. Models that fail are skipped and reported."""
    fitted, rows, skipped = {}, {}, {}
    for name, make in factories.items():
        try:
            start = time.time()
            model = make().fit(train)
            fit_s = time.time() - start
            fitted[name] = model
            rows[name] = {**evaluate_model(model, test, **eval_kwargs), "fit_seconds": round(fit_s, 1)}
            print(f"[ok] {name}: fit {fit_s:.1f}s")
        except Exception as e:
            skipped[name] = f"{type(e).__name__}: {e}"
            print(f"[skipped] {name}: {skipped[name]}")

    if "ContentBased" in fitted:
        for cf_name in ("ItemKNN", "MatrixFactorization"):
            if cf_name not in fitted:
                continue
            name = f"Hybrid({cf_name})"
            try:
                from .models.hybrid import Hybrid
                start = time.time()
                model = Hybrid(fitted[cf_name], fitted["ContentBased"], alpha=hybrid_alpha).fit(train)
                fit_s = time.time() - start
                rows[name] = {**evaluate_model(model, test, **eval_kwargs), "fit_seconds": round(fit_s, 1)}
                print(f"[ok] {name}")
            except Exception as e:
                skipped[name] = f"{type(e).__name__}: {e}"
                print(f"[skipped] {name}: {skipped[name]}")
    return pd.DataFrame(rows).T, skipped


def save_outputs(table, tag):
    RESULTS_DIR.mkdir(exist_ok=True)
    table.round(4).to_csv(RESULTS_DIR / f"results_{tag}.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: saved the table only")
        return
    for fname, cols, title in [
        (f"rating_error_{tag}.png", ["rmse", "mae"], "Rating prediction error (lower is better)"),
        (f"topn_{tag}.png", ["precision@5", "recall@5", "precision@10", "recall@10"],
         "Top-N recommendation (higher is better)"),
    ]:
        ax = table[cols].plot(kind="bar", figsize=(9, 5), rot=30, title=title)
        ax.figure.tight_layout()
        ax.figure.savefig(RESULTS_DIR / fname, dpi=150)
        plt.close(ax.figure)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val",
                   help="evaluation set; use test only for the final run")
    p.add_argument("--users", type=int, default=None, help="use only this many random users (quick mode)")
    p.add_argument("--sample-n", type=int, default=50000, help="rows used for rmse/mae")
    p.add_argument("--max-users", type=int, default=2000, help="users used for precision/recall")
    p.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight of the CF model")
    p.add_argument("--knn-k", type=int, default=40, help="ItemKNN number of neighbours")
    p.add_argument("--knn-similarity", default="pearson", help="ItemKNN similarity measure")
    p.add_argument("--skip-cold", action="store_true", help="skip the cold-start experiment")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    eval_kwargs = {"sample_n": args.sample_n, "max_users": args.max_users, "seed": args.seed}
    books = load_books()
    train, val, test = load_data(seed=args.seed)
    target = val if args.split == "val" else test
    if args.users:
        train, target = subsample_users(args.users, args.seed, train, target)
    print(f"train={len(train)} {args.split}={len(target)}")

    table, skipped = run(model_factories(books, args.knn_k, args.knn_similarity), train, target, books,
                           eval_kwargs, args.alpha)
    print(table.round(4))
    save_outputs(table, args.split)

    if not args.skip_cold:
        train_cs, test_cs = load_cold_start_split(seed=args.seed)
        if args.users:
            train_cs, test_cs = subsample_users(args.users, args.seed, train_cs, test_cs)
        print(f"cold-start: train={len(train_cs)} test={len(test_cs)}")
        cold_table, cold_skipped = run(model_factories(books, args.knn_k, args.knn_similarity),
                                       train_cs, test_cs, books, eval_kwargs, args.alpha)
        print(cold_table.round(4))
        save_outputs(cold_table, "cold_start")
        skipped.update({f"{k} (cold)": v for k, v in cold_skipped.items()})

    if skipped:
        print("\nSkipped models:")
        for name, reason in skipped.items():
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
