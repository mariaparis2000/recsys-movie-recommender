# Recommender Systems — Individual Project

Movie recommender system prototype built on the MovieLens dataset.
Developed for the Recommender Systems course at ESADE Business School (Master in Business Analytics, 2025-2026).

## Project structure

```
recommender_assignment_placeholders/
├── src/                    # Source code
│   ├── config.py           # Paths and constants
│   ├── data_loading.py     # Data loading and preprocessing
│   ├── baselines.py        # Non-personalized recommenders
│   ├── content_based.py    # Content-based filtering
│   ├── collaborative_filtering.py  # Item-item and user-user CF
│   ├── matrix_factorization.py     # SVD matrix factorization
│   └── evaluation.py       # Evaluation metrics
├── notebooks/              # Jupyter notebooks (one per method)
├── results/                # Evaluation outputs, figures, metrics
├── data/
│   ├── raw/                # Place MovieLens files here (see below)
│   └── processed/          # Generated train/test splits
├── main.py                 # Full pipeline
├── app.py                  # Streamlit interface
└── requirements.txt
```

## Dataset setup

This project uses MovieLens Latest Small. The data files are not included in the repository.

1. Download the dataset from: https://grouplens.org/datasets/movielens/latest/
2. Download ml-latest-small.zip and unzip it
3. Copy these two files into data/raw/:
   - ratings.csv — expected columns: userId, movieId, rating, timestamp
   - movies.csv — expected columns: movieId, title, genres

## Installation

```bash
pip install -r requirements.txt
pip install scikit-surprise streamlit
```

## How to run

```bash
# Full pipeline with evaluation results
python main.py

# Interactive Streamlit app
streamlit run app.py
```

## Notebooks

Run the notebooks in order from the notebooks/ folder:

1. 01_data_exploration.ipynb — EDA and dataset statistics
2. 02_baselines.ipynb — Non-personalized recommenders
3. 03_content_based.ipynb — Content-based filtering
4. 04_collaborative_filtering.ipynb — Item-item and user-user CF
5. 05_matrix_factorization.ipynb — SVD matrix factorization
6. 06_evaluation.ipynb — Full evaluation and comparison
7. 07_extensions.ipynb — TF-IDF vs raw counts, diversity, popularity bias

Each notebook imports from src/ so make sure to run from the project root.
If needed, add this cell at the top of each notebook:

```python
import sys, os
os.chdir('/path/to/recommender_assignment_placeholders')
sys.path.insert(0, '/path/to/recommender_assignment_placeholders')
```

## Models implemented

- Random (baseline)
- Most Popular
- Highest Average Rating
- Content-Based Filtering (TF-IDF on genres)
- Item-Item Collaborative Filtering
- User-User Collaborative Filtering
- Matrix Factorization (SVD via scikit-surprise)

## Evaluation results (K=10)

| Model | P@10 | NDCG@10 | Novelty | Coverage |
|---|---|---|---|---|
| Random | 0.0045 | 0.0048 | 7.66 | 0.184 |
| Most Popular | 0.2266 | 0.2604 | 1.73 | 0.006 |
| Highest Rated | 0.0683 | 0.0759 | 3.84 | 0.004 |
| Content-Based | 0.0226 | 0.0223 | 6.87 | 0.113 |
| Item-Item CF | 0.0317 | 0.0273 | 8.80 | 0.061 |
| User-User CF | 0.3367 | 0.3968 | 2.22 | 0.026 |
| Matrix Fact. (SVD) | 0.0256 | 0.0226 | 5.37 | 0.003 |
