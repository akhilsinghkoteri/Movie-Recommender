# 🎬 Movie Recommender

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-rihlhh7snu6tdjzqv8barf.streamlit.app)

A content-based movie recommendation engine. Pick a few movies you like, and get personalized suggestions based on genre similarity — no account, no sign-up, just click and try it.

**[Try it live →](https://movie-recommender-rihlhh7snu6tdjzqv8barf.streamlit.app)**

## How it works

This isn't collaborative filtering (which needs thousands of other users' ratings) — it's content-based filtering, which works off the actual properties of each movie:

1. Each movie's genres are vectorized using **TF-IDF**, which weighs rarer, more distinctive genres higher than common ones.
2. When you pick your favorite movies, their vectors are averaged into a single "taste profile."
3. **Cosine similarity** compares that profile against every movie in the catalog to find the closest matches.
4. The top-N most similar movies are returned as recommendations.

## Dataset

[MovieLens (ml-latest-small)](https://grouplens.org/datasets/movielens/latest/) from GroupLens Research — 9,742 movies.

## Tech stack

- Python
- pandas — data loading and cleaning
- scikit-learn — TF-IDF vectorization and cosine similarity
- Streamlit — web interface

## Running locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Project structure

\`\`\`
├── app.py            # Streamlit interface
├── recommender.py    # Core recommendation engine logic
├── movies.csv         # MovieLens dataset
└── requirements.txt   # Python dependencies
\`\`\`
