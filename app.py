import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("books_dataset.csv")
# Fill missing values
text_columns = ["title", "author", "genres", "description", "series_name"]
df[text_columns] = df[text_columns].fillna("")

recommendations = (
    df["title"].fillna("") + " " +
    df["author"].fillna("") + " " +
    df["genres"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["series_name"].fillna("")
).str.lower()

recommendations = recommendations.apply(
    lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x))
)

df["clean_text"] = recommendations
tfidf = TfidfVectorizer(
    max_features=2000,      # Keep top 5000 features
    ngram_range=(1, 1),     # Unigrams + Bigrams
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(df["clean_text"])

def recommend(item_name, top_n=5):
    matches = df[df["title"] == item_name]
    if matches.empty:
        return pd.DataFrame()
    idx = matches.index[0]

    # Compute similarity only for the selected book
    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    recommendations = pd.DataFrame({
        "title": df.iloc[similar_indices]["title"].values,
        "author": df.iloc[similar_indices]["author"].values,
        "similarity_score": sim_scores[similar_indices].round(4)
    })

    return recommendations

st.title("📚 Book Recommendation System")

selected_book = st.selectbox(
    "Select a Book",
    sorted(df["title"].dropna().unique())
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("Get Recommendations"):
    recommendations = recommend(selected_book, top_n)

    st.subheader("Recommended Books")

    if len(recommendations) > 0:
        st.dataframe(recommendations, use_container_width=True)
    else:
        st.warning("No recommendations found.")