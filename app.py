"""
Streamlit interface for the movie recommender.
Run locally with: streamlit run app.py
Or deploy free at https://share.streamlit.io by connecting this GitHub repo.
"""

import streamlit as st
from recommender import load_data, build_similarity_matrix, get_personalized_recommendations

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommender")
st.write("Pick a few movies you like, and get personalized recommendations based on genre similarity.")


# cache so we don't reload/rebuild the data + matrix on every interaction
@st.cache_data
def get_engine():
    movies = load_data("movies.csv")
    tfidf_matrix, similarity_matrix = build_similarity_matrix(movies)
    return movies, tfidf_matrix, similarity_matrix


movies, tfidf_matrix, similarity_matrix = get_engine()

favorite_titles = st.multiselect(
    "Select movies you like:",
    options=movies['title'].tolist(),
)

num_recs = st.slider("How many recommendations?", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    if len(favorite_titles) == 0:
        st.warning("Pick at least one movie first.")
    else:
        recommendations = get_personalized_recommendations(
            movies, tfidf_matrix, favorite_titles, n=num_recs
        )
        st.subheader("Recommended for you:")
        for i, title in enumerate(recommendations, start=1):
            st.write(f"{i}. {title}")
