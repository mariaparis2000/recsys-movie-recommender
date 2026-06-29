"""Matrix factorization using scikit-surprise SVD."""

import numpy as np
import pandas as pd
from . import config

try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import train_test_split as surprise_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False


class MatrixFactorizationRecommender:
    """Matrix factorization via SVD (scikit-surprise).

    How it works:
    SVD learns a low-dimensional representation for every user and item.
    The predicted rating for user u and item i is:

        r_hat(u, i) = mu + b_u + b_i + p_u · q_i

    where mu is the global mean, b_u and b_i are user/item biases,
    and p_u, q_i are the learned latent factor vectors.

    The model is trained by minimising the regularised squared error
    over all known ratings using stochastic gradient descent.

    Why this matters: unlike CF, SVD can generalise across the entire
    user-item space, not just items with direct co-rating overlap.
    It also handles the data sparsity problem naturally.
    """

    def __init__(self, n_factors=50, n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.model_ = None
        self.all_items_ = None
        self._ratings_train = None  # kept for fallback

    def fit(self, ratings):
        self.all_items_ = ratings[config.ITEM_COL].unique().tolist()
        self._ratings_train = ratings

        if not SURPRISE_AVAILABLE:
            print("WARNING: scikit-surprise not available — MF will return empty lists.")
            return self

        rating_min = ratings[config.RATING_COL].min()
        rating_max = ratings[config.RATING_COL].max()
        reader = Reader(rating_scale=(rating_min, rating_max))

        data = Dataset.load_from_df(
            ratings[[config.USER_COL, config.ITEM_COL, config.RATING_COL]],
            reader,
        )
        trainset = data.build_full_trainset()

        self.model_ = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            verbose=False,
        )
        self.model_.fit(trainset)
        return self

    def predict_score(self, user_id, item_id):
        if self.model_ is None:
            return 0.0
        return self.model_.predict(str(user_id), str(item_id)).est

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if self.model_ is None:
            return []

        seen = set()
        if exclude_seen:
            seen = set(ratings_train[ratings_train[config.USER_COL] == user_id][config.ITEM_COL].values)

        candidates = [it for it in self.all_items_ if it not in seen]
        scores = [(it, self.model_.predict(str(user_id), str(it)).est) for it in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scores[:n]]
