import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 레포 맨 위 폴더를 import 경로에 추가

from src.preprocess import load_data
from src.models.knn import ItemKNN

train, val, test = load_data()
m = ItemKNN().fit(train)

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
