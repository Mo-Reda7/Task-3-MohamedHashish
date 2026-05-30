import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data.csv")

# Convert text into TF-IDF vectors
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(movies["description"])

# Calculate similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend_movies(movie_name):

    idx = movies[movies["title"].str.lower() == movie_name.lower()].index

    if len(idx) == 0:
        print("Movie not found.")
        return

    idx = idx[0]

    scores = list(enumerate(similarity[idx]))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    sorted_scores = sorted_scores[1:6]

    print("\nRecommended Movies:\n")

    for movie in sorted_scores:

        movie_index = movie[0]

        print(
            movies.iloc[movie_index]["title"],
            "-> Similarity Score:",
            round(movie[1], 2)
        )

# User input
favorite_movie = input("Enter a movie you like: ")

recommend_movies(favorite_movie)