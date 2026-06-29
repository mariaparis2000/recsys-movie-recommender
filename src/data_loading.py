"""Data loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from . import config


def load_ratings(path=config.RATINGS_PATH):
    """Load user-item ratings from a CSV file.

    Expected columns: userId, movieId, rating, timestamp
    """
    df = pd.read_csv(path)
    required = [config.USER_COL, config.ITEM_COL, config.RATING_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Ratings file is missing columns: {missing}")
    return df


def load_items(path=config.ITEMS_PATH):
    """Load item metadata from a CSV file.

    Expected columns: movieId, title, genres
    """
    df = pd.read_csv(path)
    required = [config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Items file is missing columns: {missing}")
    return df


def describe_dataset(ratings, items=None):
    """Print basic dataset statistics."""
    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_items)

    print("=" * 45)
    print("DATASET SUMMARY")
    print("=" * 45)
    print(f"  Users             : {n_users:,}")
    print(f"  Items             : {n_items:,}")
    print(f"  Ratings           : {n_ratings:,}")
    print(f"  Sparsity          : {sparsity:.4%}")
    print(f"  Rating range      : {ratings[config.RATING_COL].min()} - {ratings[config.RATING_COL].max()}")
    print(f"  Mean rating       : {ratings[config.RATING_COL].mean():.2f}")

    print("\n  Rating distribution:")
    dist = ratings[config.RATING_COL].value_counts().sort_index()
    for val, count in dist.items():
        bar = "#" * int(count / n_ratings * 40)
        print(f"    {val:.1f}  {bar} ({count:,})")

    print("\n  Most active users (by rating count):")
    top_users = ratings.groupby(config.USER_COL).size().nlargest(5)
    for uid, cnt in top_users.items():
        print(f"    User {uid}: {cnt} ratings")

    print("\n  Most rated items:")
    top_items = ratings.groupby(config.ITEM_COL).size().nlargest(5)
    for iid, cnt in top_items.items():
        title = ""
        if items is not None:
            row = items[items[config.ITEM_COL] == iid]
            if not row.empty:
                title = f" — {row.iloc[0][config.TITLE_COL]}"
        print(f"    Item {iid}{title}: {cnt} ratings")

    print("=" * 45)

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
    }


def train_test_split_ratings(ratings, test_size=0.2, random_state=config.RANDOM_STATE):
    """Split ratings into train and test sets (random split)."""
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def get_seen_items(ratings, user_id):
    """Return the set of item IDs already rated by a user."""
    return set(ratings[ratings[config.USER_COL] == user_id][config.ITEM_COL].values)
