import pandas as pd
def load_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """train, val, test 평점 DataFrame 반환. 사용자별 hold-out, seed 고정."""

def load_books() -> pd.DataFrame:
    """도서 메타데이터. 컬럼: book_id, title, authors, tags(태그명 리스트), tag_counts"""
    #파일 로드
    books = pd.read_csv('data/books.csv')
    book_tags = pd.read_csv('data/book_tags.csv')
    tags = pd.read_csv('data/tags.csv')

    #노이즈 데이터 필터링
    noise_keywords = [
        'to-read', 'currently-reading', 'favorites', 'owned', 'books-i-own', 
        'library', 'audiobook', 'kindle', 'default', 'ebook', 'wish-list', 
        'did-not-finish', 'dnf', 'borrowed'
    ]
    #연산을 통해 리스트안에 단어를 한번에 검색
    valid_tags = tags[~tags['tag_name'].str.contains('|'.join(noise_keywords), case=False, na=False)]

    # 3. 책별 유효 태그 병합 및 컬럼 생성
    clean_book_tags = book_tags.merge(valid_tags, on='tag_id', how='inner')
    
    book_tags_grouped = clean_book_tags.groupby('goodreads_book_id').agg(
        tags=('tag_name', lambda x: ' '.join(x)),
        tag_counts=('tag_name', 'count')
    ).reset_index()

    #원본 데이터와 병합 및 결측치 처리
    books_df = books.merge(book_tags_grouped, on='goodreads_book_id', how='left')
    books_df['tags'] = books_df['tags'].fillna('')
    books_df['tag_counts'] = books_df['tag_counts'].fillna(0).astype(int)

    #컬럼명 5개로 구성하여 반환
    return books_df[['book_id', 'title', 'authors', 'tags', 'tag_counts']]



def load_cold_start_split(n_books: int = 500, keep: int = 5, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """선택한 도서의 평점을 keep개만 train에 남기고 나머지를 test로 보낸 (train, test) 반환"""
