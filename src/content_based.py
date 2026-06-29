"""Content-based recommender using TF-IDF on movie genres."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import config


class ContentBasedRecommender:
    """Content-based recommender using item metadata (genres).

    How it works:
    1. Each movie is represented as a TF-IDF vector built from its genres.
       TF-IDF down-weights very common genres like Drama so they don't
       dominate every recommendation.
    2. Each user gets a profile vector: a weighted sum of the item vectors
       for movies they've rated, where the weight is the rating centered
       on the user's mean. This means liked movies pull the profile
       towards their genre space and disliked ones push it away.
    3. At recommendation time we compute cosine similarity between the
       user profile and every candidate item vector, then rank.
    """

    def __init__(self, feature_col=config.GENRES_COL):
        self.feature_col = feature_col
        self.vectorizer = None
        self.item_features_ = None   # shape (n_items, n_features)
        self.item_ids_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings, items):
        # MovieLens uses '|' as genre separator — replace with spaces so
        # TfidfVectorizer treats each genre as a separate token.
        genre_text = items[self.feature_col].fillna("").str.replace("|", " ", regex=False)

        self.vectorizer = TfidfVectorizer()
        self.item_features_ = self.vectorizer.fit_transform(genre_text)  # sparse matrix

        self.item_ids_ = items[config.ITEM_COL].values
        self.item_id_to_index_ = {iid: idx for idx, iid in enumerate(self.item_ids_)}
        return self

    def build_user_profile(self, user_id, ratings_train):
        """Build a dense user profile vector from centered ratings."""
        user_ratings = ratings_train[ratings_train[config.USER_COL] == user_id]
        if user_ratings.empty:
            return None

        mean_rating = user_ratings[config.RATING_COL].mean()

        profile = np.zeros(self.item_features_.shape[1])
        weight_sum = 0.0
        for _, row in user_ratings.iterrows():
            iid = row[config.ITEM_COL]
            if iid not in self.item_id_to_index_:
                continue
            idx = self.item_id_to_index_[iid]
            centered_weight = row[config.RATING_COL] - mean_rating
            profile += centered_weight * self.item_features_[idx].toarray().flatten()
            weight_sum += abs(centered_weight)

        if weight_sum == 0:
            return None
        return profile / weight_sum

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        profile = self.build_user_profile(user_id, ratings_train)
        if profile is None:
            return []

        # Cosine similarity between profile and all item vectors
        scores = cosine_similarity([profile], self.item_features_).flatten()

        seen = set()
        if exclude_seen:
            seen = set(ratings_train[ratings_train[config.USER_COL] == user_id][config.ITEM_COL].values)

        ranked = sorted(
            ((self.item_ids_[i], scores[i]) for i in range(len(self.item_ids_)) if self.item_ids_[i] not in seen),
            key=lambda x: x[1],
            reverse=True,
        )
        return [iid for iid, _ in ranked[:n]]

    def similar_items(self, item_id, n=10):
        """Return items most similar to a given item."""
        if item_id not in self.item_id_to_index_:
            return []
        idx = self.item_id_to_index_[item_id]
        scores = cosine_similarity(self.item_features_[idx], self.item_features_).flatten()
        ranked = sorted(
            ((self.item_ids_[i], scores[i]) for i in range(len(self.item_ids_)) if self.item_ids_[i] != item_id),
            key=lambda x: x[1],
            reverse=True,
        )
        return [iid for iid, _ in ranked[:n]]
