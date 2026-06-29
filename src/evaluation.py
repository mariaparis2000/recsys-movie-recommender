"""Evaluation metrics for recommender systems.

We evaluate across three lenses (as discussed in class):
  1. Technical / offline  — ranking quality on held-out data
  2. Beyond-accuracy      — diversity, novelty, coverage
  3. (Business metrics like CTR would require an online A/B test)
"""

import numpy as np
import pandas as pd
from . import config


# ── Per-user ranking metrics ───────────────────────────────────────────────────

def precision_at_k(recommended_items, relevant_items, k=10):
    """Fraction of top-k recommendations that are relevant.

    A high Precision@K means the items we show are likely to be liked.
    Penalises recommending irrelevant items.
    """
    recs = list(recommended_items)[:k]
    if not recs:
        return 0.0
    hits = sum(1 for r in recs if r in relevant_items)
    return hits / k


def recall_at_k(recommended_items, relevant_items, k=10):
    """Fraction of all relevant items that appear in the top-k.

    A high Recall@K means we are not missing relevant items the user
    would have liked. Important when the catalog is large.
    """
    recs = list(recommended_items)[:k]
    if not relevant_items:
        return 0.0
    hits = sum(1 for r in recs if r in relevant_items)
    return hits / len(relevant_items)


def hit_rate_at_k(recommended_items, relevant_items, k=10):
    """1 if at least one relevant item appears in top-k, else 0."""
    recs = list(recommended_items)[:k]
    return float(any(r in relevant_items for r in recs))


def dcg_at_k(relevance_scores, k=10):
    """Discounted Cumulative Gain — rewards hits ranked higher."""
    scores = list(relevance_scores)[:k]
    return sum(rel / np.log2(rank + 2) for rank, rel in enumerate(scores))


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """Normalised DCG: DCG divided by the ideal DCG.

    Returns a value between 0 and 1. A score of 1 means all relevant
    items are ranked at the top in the best possible order.
    """
    recs = list(recommended_items)[:k]
    relevance = [1 if r in relevant_items else 0 for r in recs]
    actual_dcg = dcg_at_k(relevance, k)

    # Ideal: all relevant items at the top
    ideal = [1] * min(len(relevant_items), k)
    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_reciprocal_rank(recommended_items, relevant_items, k=10):
    """Reciprocal rank of the first relevant item in the top-k list.

    Useful when we care most about whether the very first recommendation
    is relevant (e.g. a single-result search).
    """
    recs = list(recommended_items)[:k]
    for rank, item in enumerate(recs, start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


# ── Beyond-accuracy metrics ────────────────────────────────────────────────────

def catalog_coverage(all_recommendations, all_items):
    """Fraction of the catalog that appears in at least one recommendation list.

    Low coverage means the system keeps recommending the same popular items
    to everyone — bad for both users and smaller-catalog items.
    """
    recommended_flat = set(item for recs in all_recommendations for item in recs)
    return len(recommended_flat) / len(set(all_items))


def novelty(recommended_items, item_popularity, n_users):
    """Mean self-information of recommended items (higher = more novel).

    Items rated by few users have high novelty score. Measures how
    surprising / non-mainstream the recommendations are.
    """
    scores = []
    for item in recommended_items:
        pop = item_popularity.get(item, 1)
        prob = pop / n_users
        scores.append(-np.log2(prob + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


# ── Full evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(model, ratings_train, ratings_test, users, k=10, items_df=None):
    """Evaluate a model over a list of users and return average metrics.

    Relevant items for a user = items in the test set with rating >= 3.5
    (i.e. items the user actually liked).
    """
    # Pre-compute item popularity for novelty
    item_popularity = ratings_train.groupby(config.ITEM_COL).size().to_dict()
    n_users_train = ratings_train[config.USER_COL].nunique()
    all_items = ratings_train[config.ITEM_COL].unique().tolist()

    results = []
    all_recs = []

    for user_id in users:
        # What the user liked in the test set
        user_test = ratings_test[ratings_test[config.USER_COL] == user_id]
        relevant = set(user_test[user_test[config.RATING_COL] >= 3.5][config.ITEM_COL].values)
        if not relevant:
            continue

        try:
            recs = model.recommend(user_id, ratings_train, n=k, exclude_seen=True)
        except Exception:
            recs = []

        all_recs.append(recs)

        results.append({
            "precision": precision_at_k(recs, relevant, k),
            "recall": recall_at_k(recs, relevant, k),
            "ndcg": ndcg_at_k(recs, relevant, k),
            "mrr": mean_reciprocal_rank(recs, relevant, k),
            "hit_rate": hit_rate_at_k(recs, relevant, k),
            "novelty": novelty(recs, item_popularity, n_users_train),
        })

    if not results:
        return {}

    coverage = catalog_coverage(all_recs, all_items)
    avg = pd.DataFrame(results).mean().to_dict()
    avg["coverage"] = coverage
    return avg
