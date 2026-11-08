"""Shared evaluation functions: RMSE, MAE, Precision@K, Recall@K."""
import numpy as np
import pandas as pd

from .models.base import Recommender


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating(model: Recommender, test: pd.DataFrame,
                    sample_n=None, seed: int = 42) -> dict:
    """Return {'rmse': ..., 'mae': ...}. Optionally evaluate on sample_n random rows."""
    if sample_n is not None and len(test) > sample_n:
        test = test.sample(sample_n, random_state=seed)
    preds = [model.predict(u, b) for u, b in zip(test["user_id"], test["book_id"])]
    return {"rmse": rmse(test["rating"], preds), "mae": mae(test["rating"], preds)}


def evaluate_topk(model: Recommender, test: pd.DataFrame, ks=(5, 10),
                  threshold: float = 4.0, max_users=None, seed: int = 42) -> dict:
    """Return {'precision@5': ..., 'recall@5': ..., 'precision@10': ..., 'recall@10': ...}.

    Test books rated >= threshold count as relevant. Users with no relevant
    book are skipped. Optionally evaluate on max_users random users.
    """
    relevant = (test[test["rating"] >= threshold]
                .groupby("user_id")["book_id"].apply(set))
    users = relevant.index.to_numpy()
    if max_users is not None and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)

    kmax = max(ks)
    prec = {k: [] for k in ks}
    rec_ = {k: [] for k in ks}
    for u in users:
        rel = relevant[u]
        recs = list(model.recommend(int(u), kmax))
        for k in ks:
            hits = len(set(recs[:k]) & rel)
            prec[k].append(hits / k)
            rec_[k].append(hits / len(rel))

    out = {}
    for k in ks:
        out[f"precision@{k}"] = float(np.mean(prec[k]))
        out[f"recall@{k}"] = float(np.mean(rec_[k]))
    return out


def evaluate_model(model: Recommender, test: pd.DataFrame, ks=(5, 10),
                   threshold: float = 4.0, sample_n=None, max_users=None,
                   seed: int = 42) -> dict:
    """Compute rmse, mae, precision@K and recall@K in one call."""
    out = evaluate_rating(model, test, sample_n=sample_n, seed=seed)
    out.update(evaluate_topk(model, test, ks=ks, threshold=threshold,
                             max_users=max_users, seed=seed))
    return out


def compare_models(models: dict, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Evaluate {name: fitted model} and return a results table (one row per model)."""
    rows = {name: evaluate_model(m, test, **kwargs) for name, m in models.items()}
    return pd.DataFrame(rows).T
