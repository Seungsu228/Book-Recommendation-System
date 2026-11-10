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

# ── 6단계 확인 ──
from src.evaluate import evaluate_topk
from src.models.baseline import Popularity

u = int(val.iloc[0].user_id)
recs = m.recommend(u, 5)
print(f"사용자 {u} 추천:", recs, "/ 이미 읽은 책과 겹침:", len(set(recs) & set(train[train.user_id == u].book_id)))
for b2 in recs:
    print("   ", books[b2])

# 비교용: 예상 평점(predict) 순으로 추천하면 어떻게 되는지
def recommend_by_predict(model, user_id, k=10):
    uid = model.uidx_[user_id]
    rated = model.R_[uid].indices
    cand = np.setdiff1d(np.arange(len(model.item_ids_)), rated)
    preds = np.array([model.predict(user_id, model.item_ids_[i]) for i in cand])
    return model.item_ids_[cand[np.argsort(-preds)[:k]]].tolist()

class ByPredict:
    def __init__(self, model): self.model = model
    def recommend(self, user_id, k=10): return recommend_by_predict(self.model, user_id, k)

print("Precision/Recall@K (val, 사용자 2,000명)")
print("  Popularity       :", evaluate_topk(Popularity().fit(train), val, max_users=2000))
print("  KNN, 예상 평점 순 (느려서 50명):", evaluate_topk(ByPredict(m), val, max_users=50))
print("  KNN, 유사도 합 순:", evaluate_topk(m, val, max_users=2000))
