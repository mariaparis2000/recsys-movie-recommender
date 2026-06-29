"""Streamlit app for the recommender system prototype."""

import streamlit as st
import pandas as pd
import numpy as np

from src import config
from src.data_loading import load_ratings, load_items, train_test_split_ratings, describe_dataset
from src.baselines import MostPopularRecommender, HighestAverageRatingRecommender, RandomRecommender
from src.content_based import ContentBasedRecommender
from src.collaborative_filtering import ItemItemCollaborativeFiltering, UserUserCollaborativeFiltering
from src.matrix_factorization import MatrixFactorizationRecommender
from src.evaluation import evaluate_model

st.set_page_config(page_title="MovieRec", page_icon="🎬", layout="wide")

# ── Load and cache everything ──────────────────────────────────────────────────

@st.cache_data
def load_data():
    ratings = load_ratings()
    items = load_items()
    train, test = train_test_split_ratings(ratings, test_size=0.2)
    return ratings, items, train, test

@st.cache_resource
def train_models(train, items):
    models = {
        "Random":             RandomRecommender(),
        "Most Popular":       MostPopularRecommender(),
        "Highest Rated":      HighestAverageRatingRecommender(min_ratings=20),
        "Content-Based":      ContentBasedRecommender(),
        "Item-Item CF":       ItemItemCollaborativeFiltering(k=20),
        "User-User CF":       UserUserCollaborativeFiltering(k=20),
        "Matrix Fact. (SVD)": MatrixFactorizationRecommender(n_factors=50, n_epochs=20),
    }
    for name, model in models.items():
        if name == "Content-Based":
            model.fit(train, items)
        else:
            model.fit(train)
    return models

# ── App layout ────────────────────────────────────────────────────────────────

st.title("🎬 MovieRec — Recommender System Prototype")
st.caption("ESADE · Recommender Systems · Individual Project")

with st.spinner("Loading data and training models (first run only)..."):
    ratings, items, train, test = load_data()
    models = train_models(train, items)

title_map = items.set_index(config.ITEM_COL)[config.TITLE_COL].to_dict()
genre_map = items.set_index(config.ITEM_COL)[config.GENRES_COL].to_dict()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Settings")

all_users = sorted(ratings[config.USER_COL].unique().tolist())
selected_user = st.sidebar.selectbox("Select a user", all_users, index=0)
n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)
selected_models = st.sidebar.multiselect(
    "Models to show",
    list(models.keys()),
    default=list(models.keys()),
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Recommendations", "Dataset Explorer", "Model Evaluation"])

# ── TAB 1: Recommendations ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Recommendations for User {selected_user}")

    # Show what this user has rated
    user_train = train[train[config.USER_COL] == selected_user].merge(items, on=config.ITEM_COL)
    user_train_display = user_train[[config.TITLE_COL, config.GENRES_COL, config.RATING_COL]].sort_values(
        config.RATING_COL, ascending=False
    )

    with st.expander(f"Training history ({len(user_train)} ratings)"):
        st.dataframe(user_train_display.head(20), use_container_width=True)

    st.markdown("---")

    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
    else:
        cols = st.columns(min(len(selected_models), 3))
        for i, name in enumerate(selected_models):
            model = models[name]
            try:
                recs = model.recommend(selected_user, train, n=n_recs, exclude_seen=True)
            except Exception as e:
                recs = []

            with cols[i % 3]:
                st.markdown(f"**{name}**")
                if not recs:
                    st.info("No recommendations.")
                else:
                    rec_data = []
                    for rank, iid in enumerate(recs, 1):
                        rec_data.append({
                            "Rank": rank,
                            "Title": title_map.get(iid, str(iid)),
                            "Genres": genre_map.get(iid, ""),
                        })
                    st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)

# ── TAB 2: Dataset Explorer ───────────────────────────────────────────────────
with tab2:
    st.subheader("Dataset Overview")

    n_users = ratings[config.USER_COL].nunique()
    n_items = ratings[config.ITEM_COL].nunique()
    n_rat = len(ratings)
    sparsity = 1 - n_rat / (n_users * n_items)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{n_users:,}")
    c2.metric("Movies", f"{n_items:,}")
    c3.metric("Ratings", f"{n_rat:,}")
    c4.metric("Sparsity", f"{sparsity:.2%}")

    st.markdown("#### Rating distribution")
    dist = ratings[config.RATING_COL].value_counts().sort_index().reset_index()
    dist.columns = ["Rating", "Count"]
    st.bar_chart(dist.set_index("Rating"))

    st.markdown("#### Most rated movies")
    top_movies = (
        ratings.groupby(config.ITEM_COL)
        .size()
        .reset_index(name="Ratings")
        .merge(items[[config.ITEM_COL, config.TITLE_COL, config.GENRES_COL]], on=config.ITEM_COL)
        .sort_values("Ratings", ascending=False)
        .head(20)
    )
    st.dataframe(top_movies[[config.TITLE_COL, config.GENRES_COL, "Ratings"]], use_container_width=True, hide_index=True)

# ── TAB 3: Evaluation ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Evaluation")
    st.info("Evaluating on a sample of 100 test users. This may take a moment.")

    results_path = config.RESULTS_DIR / "metrics.csv"

    if st.button("Run evaluation") or results_path.exists():
        if not results_path.exists():
            eval_users = test[config.USER_COL].unique()[:100]
            K = 10
            rows = []
            progress = st.progress(0)
            for idx, (name, model) in enumerate(models.items()):
                metrics = evaluate_model(model, train, test, eval_users, k=K)
                metrics["model"] = name
                rows.append(metrics)
                progress.progress((idx + 1) / len(models))

            results_df = pd.DataFrame(rows).set_index("model")
            results_df = results_df[["precision", "recall", "ndcg", "mrr", "hit_rate", "novelty", "coverage"]]
            results_df.columns = [f"P@{K}", f"R@{K}", f"NDCG@{K}", "MRR", f"HR@{K}", "Novelty", "Coverage"]
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(results_path)
        else:
            results_df = pd.read_csv(results_path, index_col=0)

        st.dataframe(results_df.round(4), use_container_width=True)

        st.markdown("#### NDCG@10 comparison")
        if f"NDCG@10" in results_df.columns:
            st.bar_chart(results_df[[f"NDCG@10"]])

        st.markdown("#### Novelty vs Precision tradeoff")
        if "Novelty" in results_df.columns and "P@10" in results_df.columns:
            scatter_data = results_df[["Novelty", "P@10"]].reset_index()
            st.dataframe(scatter_data, hide_index=True)

        st.markdown("""
        **Reading the metrics:**
        - **P@K / R@K**: how many of the top-K items are relevant; how many relevant items we found
        - **NDCG@K**: ranking quality — relevant items ranked higher score better  
        - **MRR**: how high the first relevant item appears  
        - **Novelty**: how non-mainstream the recommendations are (higher = less popular items)  
        - **Coverage**: fraction of the catalog that gets recommended across all users
        """)
