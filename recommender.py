"""
Core recommendation engine logic.
Kept separate from any interface (notebook, CLI, or web app) so the same
functions can be reused anywhere, without depending on notebook globals.
"""

import pandas as pd
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


def build_similarity_matrix(movies):
    """Vectorize genres with TF-IDF, then compute pairwise cosine similarity."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres_clean'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return tfidf_matrix, similarity_matrix


def get_recommendations(movies, similarity_matrix, title, n=10):
    """Top-N movies most similar to a single given movie."""
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


def get_personalized_recommendations(movies, tfidf_matrix, favorite_titles, n=10):
    """Top-N recommendations based on an averaged profile of several favorite movies."""
    indices = [movies[movies['title'] == t].index[0] for t in favorite_titles]
    profile_vector = tfidf_matrix[indices].mean(axis=0)
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
