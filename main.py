"""Main pipeline for the individual recommender assignment."""

import pandas as pd
import os

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings, describe_dataset
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model


def main():
    print("\n=== RECOMMENDER SYSTEMS — INDIVIDUAL PROJECT ===\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    ratings = load_ratings()
    items = load_items()

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    describe_dataset(ratings, items)

    # ── 3. Train/test split ───────────────────────────────────────────────────
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    print(f"\nTrain: {len(train):,} ratings | Test: {len(test):,} ratings\n")

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("Training models...")

    models = {
        "Random":           RandomRecommender(),
        "Most Popular":     MostPopularRecommender(),
        "Highest Rated":    HighestAverageRatingRecommender(min_ratings=20),
        "Content-Based":    ContentBasedRecommender(),
        "Item-Item CF":     ItemItemCollaborativeFiltering(k=20),
        "User-User CF":     UserUserCollaborativeFiltering(k=20),
        "Matrix Fact. (SVD)": MatrixFactorizationRecommender(n_factors=50, n_epochs=20),
    }

    for name, model in models.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        if name == "Content-Based":
            model.fit(train, items)
        else:
            model.fit(train)
        print("done")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating (sample of 200 users)...")
    eval_users = test[config.USER_COL].unique()[:200]
    K = 10

    rows = []
    for name, model in models.items():
        print(f"  Evaluating {name}...", end=" ", flush=True)
        metrics = evaluate_model(model, train, test, eval_users, k=K)
        metrics["model"] = name
        rows.append(metrics)
        print("done")

    results_df = pd.DataFrame(rows).set_index("model")
    results_df = results_df[["precision", "recall", "ndcg", "mrr", "hit_rate", "novelty", "coverage"]]
    results_df.columns = [f"P@{K}", f"R@{K}", f"NDCG@{K}", "MRR", f"HR@{K}", "Novelty", "Coverage"]

    print("\n" + "=" * 75)
    print(f"EVALUATION RESULTS  (K={K})")
    print("=" * 75)
    print(results_df.round(4).to_string())
    print("=" * 75)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    results_df.to_csv(config.RESULTS_DIR / "metrics.csv")
    print(f"\nResults saved to {config.RESULTS_DIR / 'metrics.csv'}")

    # ── 6. Recommendation examples ────────────────────────────────────────────
    sample_users = list(eval_users[:3])
    print("\n" + "=" * 75)
    print("RECOMMENDATION EXAMPLES (top-5 per model, 3 users)")
    print("=" * 75)

    title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()

    for uid in sample_users:
        print(f"\n--- User {uid} ---")
        for name, model in models.items():
            try:
                recs = model.recommend(uid, train, n=5, exclude_seen=True)
            except Exception:
                recs = []
            titles = [title_map.get(r, str(r)) for r in recs]
            print(f"  {name}:")
            for t in titles:
                print(f"    • {t}")

    print("\nDone. Run `streamlit run app.py` to explore the results interactively.")


if __name__ == "__main__":
    main()
