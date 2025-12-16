import streamlit as st
import numpy as np
import pandas as pd
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
st.write("Get similar articles using NLP-based similarity (Word2Vec embeddings).")

# ----------------------------------
# Load Data
# ----------------------------------
@st.cache_resource
def load_data():
    # Load articles from JSON
    df = pd.read_json("data/articles.json")

    # Load embeddings
    embeddings = np.load("data/embeddings_w2v.npy")

    return df, embeddings


df, embeddings = load_data()

# ----------------------------------
# Recommendation Function
# ----------------------------------
def recommend(article_id, top_n=3):
    if article_id >= len(embeddings):
        return None, None

    query_vec = embeddings[article_id].reshape(1, -1)
    similarities = cosine_similarity(query_vec, embeddings)[0]

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
    with st.spinner("Computing similarities..."):
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
