# Data

The raw data is **not tracked** in this repository because of its size
(`ratings.csv` alone is about 69MB). Download it and place the files in this folder.

## Download

Source: [Goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) by Zygmunt Zając

Either:

- Download each file from the repository root, or
- Download the zipped files from the [releases page](https://github.com/zygmuntz/goodbooks-10k/releases) and unzip them here.

## Expected layout

```
data/
├── README.md        # this file (the only file tracked by git)
├── ratings.csv      # user_id, book_id, rating (1–5)
├── books.csv        # book metadata (title, authors, year, average rating, ...)
├── book_tags.csv    # tags assigned to each book, with counts
├── tags.csv         # tag id → tag name
└── to_read.csv      # user_id, book_id pairs marked "to read"
```

## Notes

- `ratings.csv` and `to_read.csv` use `book_id` (1–10,000).
- `book_tags.csv` uses `goodreads_book_id`. Map it to `book_id` through `books.csv`.
- Everything in `data/` except this README is ignored by git (see `.gitignore`).
  Do not force-add data files.
