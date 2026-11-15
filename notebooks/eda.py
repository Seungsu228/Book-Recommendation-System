"""Exploratory data analysis for Goodbooks-10k.

Run from the repo root:
    python3 notebooks/eda.py

Prints summary statistics and saves figures and tables to results/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocess import DATA_DIR, load_books, load_cold_start_split, load_data, load_ratings

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"saved results/{name}")


def log_hist(values, title, xlabel, name):
    bins = np.logspace(0, np.log10(values.max()), 40)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=bins, color="#4C78A8")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.axvline(values.median(), color="#E45756", linestyle="--", label=f"median = {values.median():.0f}")
    ax.legend()
    save(fig, name)


def main():
    ratings = load_ratings()
    books = load_books()

    # 1. Overview
    n_users, n_books, n_ratings = ratings["user_id"].nunique(), ratings["book_id"].nunique(), len(ratings)
    density = n_ratings / (n_users * n_books)
    print(f"users={n_users:,} books={n_books:,} ratings={n_ratings:,} density={density:.2%}")

    # 2. Rating distribution
    dist = ratings["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(dist.index.astype(str), dist.values, color="#4C78A8")
    for bar, v in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v / n_ratings:.1%}", ha="center", va="bottom")
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    save(fig, "eda_rating_distribution.png")
    print(f"mean rating={ratings['rating'].mean():.2f}")

    # 3. Ratings per user and per book
    per_user = ratings.groupby("user_id").size()
    per_book = ratings.groupby("book_id").size()
    log_hist(per_user, "Ratings per user", "Number of ratings (log scale)", "eda_ratings_per_user.png")
    log_hist(per_book, "Ratings per book", "Number of ratings (log scale)", "eda_ratings_per_book.png")
    top10pct = per_book.sort_values(ascending=False).head(int(len(per_book) * 0.1)).sum() / n_ratings
    print(f"ratings per user: median={per_user.median():.0f} mean={per_user.mean():.1f} max={per_user.max()}")
    print(f"ratings per book: median={per_book.median():.0f} mean={per_book.mean():.1f} max={per_book.max()}")
    print(f"top 10% of books receive {top10pct:.1%} of all ratings")

    # 4. Most rated books
    top = (per_book.sort_values(ascending=False).head(10).rename("n_ratings").reset_index()
           .merge(books[["book_id", "title", "authors"]], on="book_id"))
    top["mean_rating"] = top["book_id"].map(ratings.groupby("book_id")["rating"].mean()).round(2)
    top.to_csv(OUT / "eda_top_books.csv", index=False)
    print(top[["title", "authors", "n_ratings", "mean_rating"]].to_string(index=False))

    # 5. Tags before and after cleaning
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    book_tags = pd.read_csv(DATA_DIR / "book_tags.csv")
    raw = (book_tags[book_tags["count"] > 0].merge(tags, on="tag_id")
           .groupby("tag_name")["count"].sum().sort_values(ascending=False).head(15))
    cleaned = {}
    for names, counts in zip(books["tags"], books["tag_counts"]):
        for n, c in zip(names, counts):
            cleaned[n] = cleaned.get(n, 0) + c
    cleaned = pd.Series(cleaned).sort_values(ascending=False).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, series, title in [(axes[0], raw, "Top tags before cleaning"),
                              (axes[1], cleaned, "Top tags after cleaning")]:
        ax.barh(series.index[::-1], series.values[::-1], color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("Tag count")
    save(fig, "eda_top_tags.png")
    n_tags = books["tags"].apply(len)
    print(f"tags per book after cleaning: mean={n_tags.mean():.1f} min={n_tags.min()} max={n_tags.max()}")

    # 6. Splits
    train, val, test = load_data()
    train_cs, test_cs = load_cold_start_split()
    split = pd.DataFrame({
        "split": ["train", "val", "test", "cold-start train", "cold-start test"],
        "ratings": [len(train), len(val), len(test), len(train_cs), len(test_cs)],
        "users": [d["user_id"].nunique() for d in (train, val, test, train_cs, test_cs)],
        "books": [d["book_id"].nunique() for d in (train, val, test, train_cs, test_cs)],
    })
    split.to_csv(OUT / "eda_split_sizes.csv", index=False)
    print(split.to_string(index=False))


if __name__ == "__main__":
    main()
