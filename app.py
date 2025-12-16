import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# App Config
# ----------------------------------
st.set_page_config(
    page_title="NLP Article Recommender",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Article Recommendation System")
st.write("Article similarity using TF-IDF and cosine similarity.")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_resource
def load_data():
    df = pd.read_json("data/articles.json")
    return df


df = load_data()

# ----------------------------------
# Vectorization
# ----------------------------------
@st.cache_resource
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    vectors = vectorizer.fit_transform(corpus)
    return vectors


tfidf_matrix = build_tfidf(df["content"])

# ----------------------------------
# Recommendation Function
# ----------------------------------
def recommend(article_id, top_n=3):
    if article_id >= len(df):
        return None, None

    similarities = cosine_similarity(
        tfidf_matrix[article_id],
        tfidf_matrix
    ).flatten()

    top_indices = similarities.argsort()[::-1][1 : top_n + 1]

    results = []
    for idx in top_indices:
        results.append({
            "ID": int(idx),
            "Title": df.iloc[idx]["title"],
            "Score": float(similarities[idx])
        })

    return df.iloc[article_id]["title"], results


# ----------------------------------
# User Inputs
# ----------------------------------
article_id = st.number_input(
    "Article ID",
    min_value=0,
    max_value=len(df) - 1,
    step=1
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=3
)

# ----------------------------------
# Action
# ----------------------------------
if st.button("🔍 Get Recommendations"):
    with st.spinner("Computing similarity..."):
        query_title, recommendations = recommend(article_id, top_n)

        if query_title is None:
            st.error("Invalid Article ID")
        else:
            st.subheader("📄 Reference Article")
            st.write(query_title)

            st.subheader("✨ Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(
                    f"""
                    **{i}. {rec['Title']}**  
                    • ID: `{rec['ID']}`  
                    • Similarity Score: `{rec['Score']:.4f}`
                    """
                )
