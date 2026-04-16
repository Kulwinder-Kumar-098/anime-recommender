# ===============================
# ANIME RECOMMENDER PIPELINE
# ===============================
# Content-Based Recommendation System
# Uses: CountVectorizer + Cosine Similarity
# (No traditional ML model training)
# ===============================
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


# ===============================
# 2. DATA INGESTION
# ===============================
def load_data(path):
    anime = pd.read_csv(os.path.join(path, 'anime.csv'))
    rating = pd.read_csv(os.path.join(path, 'rating.csv'))
    return anime, rating


# ===============================
# 3. PREPROCESSING
# ===============================
def preprocess(anime):
    anime['rating'] = anime['rating'].fillna(anime['rating'].mean())
    anime = anime.dropna()

    anime['tags'] = anime['genre'] + "," + anime['type']

    return anime[['anime_id', 'name', 'tags']]


# ===============================
# 4. TEXT PROCESSING
# ===============================
ps = PorterStemmer()

def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])


def apply_stemming(df):
    df['tags'] = df['tags'].apply(stem_text)
    return df


# ===============================
# 5. FEATURE ENGINEERING (VECTORIZATION)
# ===============================
# Convert anime genres/tags to numerical vectors using CountVectorizer
def vectorize(df):
    cv = CountVectorizer(max_features=5000)
    vectors = cv.fit_transform(df['tags']).toarray()
    return cv, vectors


# ===SIMILARITY COMPUTATION (Cosine Similarity)
# ===============================
# Calculate cosine similarity between all anime vectors
# No model training - just mathematical similarity scores
# ===============================
def build_similarity(vectors):
    return cosine_similarity(vectors)


# ===============================
# 7. RECOMMENDATION ENGINE
# ===============================
def recommend(user_input, df, similarity, cv, vectors, top_n=5):

    matched = df[df['name'].str.lower() == user_input.lower()]

    if not matched.empty:
        idx = matched.index[0]
        distances = similarity[idx]
        results = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:top_n+1]

    else:
        stemmed = stem_text(user_input.lower())
        input_vec = cv.transform([stemmed]).toarray()
        distances = cosine_similarity(input_vec, vectors)[0]

        results = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[:top_n]

    return [df.iloc[i[0]]['name'] for i in results]



# 8. SAVE / LOAD PIPELINE
# ===============================
def save_pipeline(df, similarity, cv, path="model.pkl"):
    with open(path, 'wb') as f:
        pickle.dump((df, similarity, cv), f)


def load_pipeline(path="model.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ===============================
# Build the recommendation system from data
# (Not model training - just vectorization + similarity computation)
# 9. MAIN PIPELINE
# ===============================
def run_pipeline(data_path):

    anime, rating = load_data(data_path)

    df = preprocess(anime)
    df = apply_stemming(df)

    cv, vectors = vectorize(df)
    similarity = build_similarity(vectors)

    save_pipeline(df, similarity, cv)

    return df, similarity, cv, vectors


# ===============================
# 10. EXECUTION
# ===============================
if __name__ == "__main__":
    data_path = "your_dataset_path"

    df, similarity, cv, vectors = run_pipeline(data_path)

    print(recommend("Naruto", df, similarity, cv, vectors))
    print(recommend("space sci-fi thriller", df, similarity, cv, vectors))
