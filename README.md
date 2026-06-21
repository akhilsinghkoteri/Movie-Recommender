A content-based movie recommendation engine. Pick a few movies you like, and get personalized suggestions based on genre similarity — no account, no sign-up, just click and try it.

Try it live →

How it works

This isn't collaborative filtering (which needs thousands of other users' ratings) — it's content-based filtering, which works off the actual properties of each movie:


Each movie's genres are vectorized using TF-IDF, which weighs rarer, more distinctive genres higher than common ones.
When you pick your favorite movies, their vectors are averaged into a single "taste profile."
Cosine similarity compares that profile against every movie in the catalog to find the closest matches.
The top-N most similar movies are returned as recommendations.


Dataset

MovieLens (ml-latest-small) from GroupLens Research — 9,742 movies.

Tech stack


Python
pandas — data loading and cleaning
scikit-learn — TF-IDF vectorization and cosine similarity
Streamlit — web interface


Running locally

bashpip install -r requirements.txt
streamlit run app.py

Project structure

├── app.py            # Streamlit interface
├── recommender.py    # Core recommendation engine logic
├── movies.csv         # MovieLens dataset
└── requirements.txt   # Python dependencies
