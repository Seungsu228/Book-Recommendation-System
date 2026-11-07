"""Data loading, preprocessing, and train/val/test splitting for Goodbooks-10k."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 42

STOP_TAGS = {
    "to-read", "currently-reading", "read", "reading", "favorites", "favourites",
    "favorite", "owned", "owned-books", "books-i-own", "my-books", "my-library",
    "to-buy", "wish-list", "wishlist", "default", "library", "kindle", "ebook",
    "ebooks", "audiobook", "audiobooks", "audible", "re-read", "dnf",
    "did-not-finish", "abandoned", "unfinished", "tbr", "i-own", "own-it","have",
    "borrowed", "e-book", "book-club", "favorites-", "all-time-favorites",
}
NOISE_TAG_PATTERN = (
    r"\d+|\d{4}-?reads?|\d{4}-books?|read-in-\d{4}|\d+-stars?"
    r"|.*favou?rite.*|.*to-read.*|.*currently-reading.*|(my-)?books?|novels?|books-i-own"
    r"|read-?\d{4}|re-?reads?|faves?|audio.*|.*owned.*|.*kindle.*|.*e-?books?.*|.*library.*"
)


def load_ratings(data_dir=DATA_DIR) -> pd.DataFrame:
    """Read ratings.csv and drop duplicate (user_id, book_id) pairs."""
    df = pd.read_csv(Path(data_dir) / "ratings.csv")
    df = df[["user_id", "book_id", "rating"]]
    return df.drop_duplicates(["user_id", "book_id"]).reset_index(drop=True)


def load_data(seed: int = SEED, data_dir=DATA_DIR, val_ratio=0.1,
              test_ratio=0.1, min_ratings=5):
    """Return (train, val, test) rating DataFrames using a per-user hold-out.

    Each user's ratings are shuffled, then test_ratio and val_ratio of them
    (at least one each) are held out. Users with fewer than min_ratings
    ratings stay entirely in train. The split is fixed by seed.
    """
    ratings = load_ratings(data_dir)
    shuffled = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    grp = shuffled.groupby("user_id")["rating"]
    n = grp.transform("size").to_numpy()
    rank = grp.cumcount().to_numpy()

    n_test = np.maximum(1, np.round(n * test_ratio)).astype(int)
    n_val = np.maximum(1, np.round(n * val_ratio)).astype(int)
    eligible = n >= min_ratings

    is_test = eligible & (rank < n_test)
    is_val = eligible & ~is_test & (rank < n_test + n_val)

    train = shuffled[~(is_test | is_val)].reset_index(drop=True)
    val = shuffled[is_val].reset_index(drop=True)
    test = shuffled[is_test].reset_index(drop=True)
    return train, val, test


def load_books(data_dir=DATA_DIR, max_tags: int = 30) -> pd.DataFrame:
    """Return book metadata with columns: book_id, title, authors, tags, tag_counts.

    tags and tag_counts are lists of the same length and order (most used first).
    Personal-shelf tags such as 'to-read' are removed and at most max_tags
    tags are kept per book.
    """
    data_dir = Path(data_dir)
    books = pd.read_csv(data_dir / "books.csv")
    books = books[["book_id", "goodreads_book_id", "title", "authors"]].copy()

    tags = pd.read_csv(data_dir / "tags.csv")
    book_tags = pd.read_csv(data_dir / "book_tags.csv")
    # book_tags.csv uses goodreads_book_id, so map it to book_id through books.csv.
    bt = (book_tags[book_tags["count"] > 0]
          .merge(tags, on="tag_id")
          .merge(books[["book_id", "goodreads_book_id"]], on="goodreads_book_id"))
    bt = bt[~bt["tag_name"].isin(STOP_TAGS)]
    bt = bt[~bt["tag_name"].str.fullmatch(NOISE_TAG_PATTERN)]
    bt = (bt.sort_values(["book_id", "count"], ascending=[True, False])
            .groupby("book_id").head(max_tags))

    names = {bid: d["tag_name"].tolist() for bid, d in bt.groupby("book_id")}
    counts = {bid: d["count"].tolist() for bid, d in bt.groupby("book_id")}
    books["tags"] = books["book_id"].map(lambda b: names.get(b, []))
    books["tag_counts"] = books["book_id"].map(lambda b: counts.get(b, []))
    return books.drop(columns=["goodreads_book_id"]).reset_index(drop=True)


def load_cold_start_split(n_books: int = 500, keep: int = 5, seed: int = SEED,
                          data_dir=DATA_DIR, min_ratings: int = 100):
    """Return (train, test) where selected books keep only `keep` ratings in train.

    n_books books with at least min_ratings ratings are chosen at random.
    For each chosen book, `keep` ratings stay in train and the rest go to test.
    All ratings of the other books stay in train.
    """
    ratings = load_ratings(data_dir)
    rng = np.random.default_rng(seed)

    counts = ratings["book_id"].value_counts()
    eligible = np.sort(counts[counts >= max(min_ratings, keep + 1)].index.to_numpy())
    n_books = min(n_books, len(eligible))
    cold_books = rng.choice(eligible, size=n_books, replace=False)

    cold = ratings[ratings["book_id"].isin(cold_books)].sample(frac=1.0, random_state=seed)
    keep_idx = cold.groupby("book_id").head(keep).index
    test = cold.drop(index=keep_idx)
    train = ratings.drop(index=test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)
