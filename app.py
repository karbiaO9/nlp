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
st.write("Get similar articles using NLP-based similarity (JSON + local embeddings).")

# ----------------------------------
# Load Data (JSON + embeddings)
# ----------------------------------
@st.cache_resource

def load_data():
    # Load articles from JSON
    df = pd.read_json("data/articles.json")

    # Keep only required columns
    df = df[["ID", "Title"]]

    # Load embeddings
    embeddings = np.load("data/embeddings_w2v.npy")

    # Safety check
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Mismatch: {len(df)} articles but {embeddings.shape[0]} embeddings"
        )

    df["_index"] = range(len(df))
    df = df.set_index("ID")

    return df, embeddings

try:
    df, embeddings = load_data()
    st.success("✅ Data and embeddings loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# ----------------------------------
# Recommender Function
# ----------------------------------

def recommend(article_id, top_n=5):
    article_id = int(article_id)

    if article_id not in df.index:
        return []

    idx = df.loc[article_id, "_index"]
    query_vec = embeddings[idx].reshape(1, -1)

    scores = cosine_similarity(query_vec, embeddings)[0]
    ranked_indices = scores.argsort()[::-1]

    results = []
    for i in ranked_indices:
        result_id = df[df["_index"] == i].index[0]

        if result_id == article_id:
            continue

        results.append({
            "ID": int(result_id),
            "Title": df.loc[result_id, "Title"],
            "Score": float(scores[i])
        })

        if len(results) >= top_n:
            break

    return results

# ----------------------------------
# User Inputs
# ----------------------------------
article_id = st.number_input(
    "Article ID",
    min_value=int(df.index.min()),
    max_value=int(df.index.max()),
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
    with st.spinner("Computing recommendations..."):
        results = recommend(article_id, top_n)

        if not results:
            st.warning("No recommendations found for this article ID.")
        else:
            st.subheader("📄 Reference Article")
            st.write(df.loc[int(article_id), "Title"])

            st.subheader("✨ Recommendations")
            for i, rec in enumerate(results, 1):
                st.markdown(
                    f"""
                    **{i}. {rec['Title']}**  
                    • ID: `{rec['ID']}`  
                    • Similarity Score: `{rec['Score']:.4f}`
                    """
                )
