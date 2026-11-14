import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.preprocess import load_data, load_books
from src.evaluate import evaluate_rating, evaluate_topk
from src.models.knn import ItemKNN
from src.models.content import ContentBased
from src.models.hybrid import Hybrid

def main():
    print("데이터를 로드하는 중입니다...")
    train, val, _ = load_data()
    books = load_books()
    
    knn_model = ItemKNN(k=40, similarity="pearson")
    content_model = ContentBased(books=books)
    
    # 1. 모델 조립
    model = Hybrid(cf_model=knn_model, content_model=content_model, alpha=0.5)
    
    print("\n[1단계] 기반 모델(KNN, Content) 사전 학습을 시작합니다...")
    t_fit_start = time.time()
    model.fit(train)
    print(f"사전 학습 완료! ({time.time() - t_fit_start:.0f}초 소요)\n")
    
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("[2단계] 최적의 alpha 값을 찾기 위한 평가를 시작합니다...")
    results = []
    
    for alpha in alphas:
        print(f"  -> alpha={alpha:.1f} 평가 중... ", end='', flush=True)
        
        model.alpha = alpha
        
        t_eval_start = time.time()
        rmse_res = evaluate_rating(model, val)
        p10_res = evaluate_topk(model, val)
        eval_time = time.time() - t_eval_start
        
        # 딕셔너리로 반환될 경우 숫자값(float)만 안전하게 추출 (괄호 오타 수정 완료)
        rmse = rmse_res.get('rmse', list(rmse_res.values())[0]) if isinstance(rmse_res, dict) else rmse_res
        p10 = p10_res.get('precision@10', p10_res.get('p@10', list(p10_res.values())[0])) if isinstance(p10_res, dict) else p10_res
        
        print(f"완료! (rmse={rmse:.4f} / p@10={p10:.4f} / {eval_time:.0f}초 소요)")
        results.append({'alpha': alpha, 'rmse': rmse, 'p@10': p10})

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/hybrid_tuning.csv", index=False)
    print("\n튜닝 완료! 결과가 results/hybrid_tuning.csv에 저장되었습니다.")

if __name__ == "__main__":
    main()