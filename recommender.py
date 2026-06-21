"""
Core recommendation engine logic.
Kept separate from any interface (notebook, CLI, or web app) so the same
functions can be reused anywhere, without depending on notebook globals.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path="movies.csv"):
    """Load the movies CSV and clean the genres column for vectorizing."""
    movies = pd.read_csv(path)
    movies['genres_clean'] = (
        movies['genres']
        .str.replace('|', ' ', regex=False)
        .str.replace('-', '', regex=False)
    )
    return movies


def build_tfidf_matrix(movies):
    """Vectorize genres with TF-IDF. No full similarity matrix is precomputed
    here on purpose -- a 9742x9742 dense matrix takes ~725MB of RAM, which
    blows past free hosting tier memory limits. We compute similarity rows
    on demand instead, only for the movie(s) actually being looked up."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    return tfidf_matrix


def get_recommendations(movies, tfidf_matrix, title, n=10):
    """Top-N movies most similar to a single given movie."""
    idx = movies[movies['title'] == title].index[0]
    # compare just this one movie's vector against all others -- a single
    # row, not the full matrix
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores_idx = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_scores_idx = sim_scores_idx[1:n + 1]
    movie_indices = [i[0] for i in sim_scores_idx]
    return movies['title'].iloc[movie_indices].tolist()


def get_personalized_recommendations(movies, tfidf_matrix, favorite_titles, n=10):
    """Top-N recommendations based on an averaged profile of several favorite movies."""
    indices = [movies[movies['title'] == t].index[0] for t in favorite_titles]
    profile_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
    sim_scores = cosine_similarity(profile_vector, tfidf_matrix).flatten()
    sim_scores_idx = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in sim_scores_idx:
        t = movies['title'].iloc[idx]
        if t not in favorite_titles:
            recommendations.append(t)
        if len(recommendations) == n:
            break
    return recommendations
