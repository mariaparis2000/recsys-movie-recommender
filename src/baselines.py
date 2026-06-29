"""Non-personalized baseline recommenders."""

import numpy as np
import pandas as pd
from . import config


class MostPopularRecommender:
    """Recommend the items that have been rated most often.

    No personalization — every user gets the same list (minus what they've
    already seen). This is a strong baseline in practice because popular
    items tend to be well-known and broadly liked.
    """

    def __init__(self):
        self.ranking_ = None

    def fit(self, ratings, items=None):
        counts = ratings.groupby(config.ITEM_COL).size().reset_index(name="count")
        self.ranking_ = counts.sort_values("count", ascending=False)[config.ITEM_COL].tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        recs = [item for item in self.ranking_ if item not in seen]
        return recs[:n]


class HighestAverageRatingRecommender:
    """Recommend items with the highest average rating.

    We require a minimum number of ratings to avoid recommending obscure
    items that happen to have a perfect 5.0 from a single user.
    """

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings
        self.ranking_ = None

    def fit(self, ratings, items=None):
        stats = ratings.groupby(config.ITEM_COL)[config.RATING_COL].agg(["mean", "count"])
        filtered = stats[stats["count"] >= self.min_ratings]
        self.ranking_ = filtered.sort_values("mean", ascending=False).index.tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        recs = [item for item in self.ranking_ if item not in seen]
        return recs[:n]


class RandomRecommender:
    """Recommend random unseen items.

    Useful as a lower-bound baseline to confirm that our models are
    actually learning something meaningful.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.items_ = None

    def fit(self, ratings, items=None):
        self.items_ = ratings[config.ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        rng = np.random.RandomState(self.random_state)
        seen = get_seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [i for i in self.items_ if i not in seen]
        n = min(n, len(candidates))
        return rng.choice(candidates, size=n, replace=False).tolist()


# ── helper reused across this module ──────────────────────────────────────────

def get_seen_items(ratings, user_id):
    return set(ratings[ratings[config.USER_COL] == user_id][config.ITEM_COL].values)
