"""Collaborative filtering: item-item and user-user."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from . import config


def _get_seen(ratings_train, user_id):
    return set(ratings_train[ratings_train[config.USER_COL] == user_id][config.ITEM_COL].values)


class ItemItemCollaborativeFiltering:
    """Item-item collaborative filtering.

    Idea: two items are similar if users tend to rate them the same way.
    To recommend for a user, we score each unseen item by looking at
    how similar it is to the items the user has already rated,
    weighted by those ratings.

    Formula for predicted score of item i for user u:
        score(u, i) = sum_j [sim(i,j) * r(u,j)] / sum_j |sim(i,j)|
    where j ranges over the top-k most similar items already rated by u.
    """

    def __init__(self, k=20, similarity="cosine"):
        self.k = k
        self.similarity = similarity
        self.user_item_matrix_ = None
        self.item_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        self.user_ids_ = ratings[config.USER_COL].unique()
        self.item_ids_ = ratings[config.ITEM_COL].unique()
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}

        row = ratings[config.USER_COL].map(self.user_id_to_index_)
        col = ratings[config.ITEM_COL].map(self.item_id_to_index_)
        data = ratings[config.RATING_COL].values

        self.user_item_matrix_ = csr_matrix(
            (data, (row, col)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )
        # Item-item similarity: transpose so items are rows
        item_matrix = self.user_item_matrix_.T
        self.item_similarity_ = cosine_similarity(item_matrix, dense_output=True)
        return self

    def predict_score(self, user_id, item_id):
        if user_id not in self.user_id_to_index_ or item_id not in self.item_id_to_index_:
            return 0.0

        u_idx = self.user_id_to_index_[user_id]
        i_idx = self.item_id_to_index_[item_id]

        # Similarity vector for target item against all items
        sim_vector = self.item_similarity_[i_idx]

        # Items the user has rated
        user_row = self.user_item_matrix_[u_idx].toarray().flatten()
        rated_mask = user_row > 0

        if rated_mask.sum() == 0:
            return 0.0

        rated_sims = sim_vector[rated_mask]
        rated_ratings = user_row[rated_mask]

        # Take only the top-k neighbours
        top_k_idx = np.argsort(rated_sims)[::-1][: self.k]
        top_sims = rated_sims[top_k_idx]
        top_ratings = rated_ratings[top_k_idx]

        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return 0.0
        return float(np.dot(top_sims, top_ratings) / denom)

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = _get_seen(ratings_train, user_id) if exclude_seen else set()
        candidates = [it for it in self.item_ids_ if it not in seen]

        scores = [(it, self.predict_score(user_id, it)) for it in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scores[:n]]


class UserUserCollaborativeFiltering:
    """User-user collaborative filtering (extension).

    Instead of item similarity, we find users who are most similar to the
    target user and borrow their ratings to predict unseen items.
    """

    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix_ = None
        self.user_similarity_ = None
        self.user_ids_ = None
        self.item_ids_ = None
        self.user_id_to_index_ = None
        self.item_id_to_index_ = None

    def fit(self, ratings):
        self.user_ids_ = ratings[config.USER_COL].unique()
        self.item_ids_ = ratings[config.ITEM_COL].unique()
        self.user_id_to_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_id_to_index_ = {it: i for i, it in enumerate(self.item_ids_)}

        row = ratings[config.USER_COL].map(self.user_id_to_index_)
        col = ratings[config.ITEM_COL].map(self.item_id_to_index_)
        data = ratings[config.RATING_COL].values

        self.user_item_matrix_ = csr_matrix(
            (data, (row, col)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
        )
        self.user_similarity_ = cosine_similarity(self.user_item_matrix_, dense_output=True)
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if user_id not in self.user_id_to_index_:
            return []

        u_idx = self.user_id_to_index_[user_id]
        sim_vector = self.user_similarity_[u_idx].copy()
        sim_vector[u_idx] = 0  # exclude self

        # Top-k similar users
        top_k_users = np.argsort(sim_vector)[::-1][: self.k]
        seen = _get_seen(ratings_train, user_id) if exclude_seen else set()

        # Weighted sum of neighbour ratings for each candidate item
        item_scores = {}
        for nb_idx in top_k_users:
            sim = sim_vector[nb_idx]
            if sim <= 0:
                continue
            nb_row = self.user_item_matrix_[nb_idx].toarray().flatten()
            for i_idx, rating in enumerate(nb_row):
                if rating == 0:
                    continue
                item_id = self.item_ids_[i_idx]
                if item_id in seen:
                    continue
                item_scores[item_id] = item_scores.get(item_id, 0.0) + sim * rating

        ranked = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [it for it, _ in ranked[:n]]
