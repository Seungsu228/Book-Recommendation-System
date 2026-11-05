from .base import Recommender


class MatrixFactorization(Recommender):
    def __init__(self, n_factors: int = 50, lr: float = 0.005, reg: float = 0.02,
                 n_epochs: int = 20, seed: int = 42):
        """
        n_factors: latent factor 수
        lr: SGD learning rate
        reg: L2 regularization 강도 (lambda)
        n_epochs: 학습 반복 횟수
        seed: 초기화 랜덤 시드
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed
