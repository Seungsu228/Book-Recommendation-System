import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 레포 맨 위 폴더를 import 경로에 추가

from src.preprocess import load_data
import numpy as np
from src.models.knn import ItemKNN

train, val, test = load_data()
t0 = time.time()
m = ItemKNN().fit(train)
print(f"fit 시간: {time.time() - t0:.1f}초")

# ── 1~2단계 확인 ──
print("R_ shape:", m.R_.shape)            # (사용자 수, 도서 수)
print("R_ nnz:", m.R_.nnz, "/ train:", len(train))  # 두 값이 같아야 함
u, b = train.iloc[0]["user_id"], train.iloc[0]["book_id"]
print("예시:", int(u), int(b), "평점", train.iloc[0]["rating"], "→ R_ 값", m.R_[m.uidx_[u], m.iidx_[b]])

# ── 3단계 확인 ──
print("전체 평균:", round(m.global_mean_, 4), "/ pandas로 계산:", round(train["rating"].mean(), 4))
b_mean = train.groupby("book_id")["rating"].mean()
print("도서", int(b), "평균:", round(float(m.item_mean_[m.iidx_[b]]), 4), "/ pandas로 계산:", round(b_mean[b], 4))
print("가장 인기 있는 도서 3권:", m.item_ids_[m.popular_[:3]].tolist())

# ── 4단계 확인 ──
import pandas as pd
books = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "books.csv").set_index("book_id")["title"]
print("저장된 이웃 수:", m.S_.nnz, "(도서 1만 권 x k=20 이하)")
q = 2  # Harry Potter and the Sorcerer's Stone
row = m.S_[m.iidx_[q]]
order = np.argsort(-row.data)[:5]
print(f"'{books[q]}'와 비슷한 책:")
for j, s in zip(row.indices[order], row.data[order]):
    print(f"  {s:.3f}  {books[m.item_ids_[j]]}")

# ── 5단계 확인 ──
from src.evaluate import evaluate_rating
from src.models.baseline import BookMean
row = val.iloc[0]
print("val 첫 줄:", int(row.user_id), int(row.book_id), "실제", row.rating, "→ 예측", round(m.predict(row.user_id, row.book_id), 3))
print("처음 보는 사용자/도서:", round(m.predict(10**9, 2), 3), round(m.predict(int(row.user_id), 10**9), 3))
print("RMSE/MAE (val 5만 건)")
print("  BookMean:", evaluate_rating(BookMean().fit(train), val, sample_n=50000))
print("  ItemKNN :", evaluate_rating(m, val, sample_n=50000))
