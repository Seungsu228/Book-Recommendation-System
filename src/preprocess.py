import pandas as pd
def load_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train, val, test 평점 DataFrame 반환. 사용자별 hold-out, seed 고정."""

def load_books() -> pd.DataFrame:
    """도서 메타데이터. 컬럼: book_id, title, authors, tags(태그명 리스트), tag_counts"""

def load_cold_start_split(n_books: int = 500, keep: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """선택한 도서의 평점을 keep개만 train에 남기고 나머지를 test로 보낸 (train, test) 반환"""
