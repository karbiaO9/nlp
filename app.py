import streamlit as st
import requests

# ----------------------------------
# App Config
# ----------------------------------
st.set_page_config(
    page_title="NLP Article Recommender",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Article Recommendation System")
st.write("Get similar articles using NLP-based similarity.")

# ----------------------------------
# API Configuration
# ----------------------------------
API_BASE_URL = "https://nlpapi-aynb.onrender.com/"  # change only if deployed elsewhere

# ----------------------------------
# User Inputs
# ----------------------------------
article_id = st.number_input(
    "Article ID",
    min_value=0,
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
    with st.spinner("Fetching recommendations..."):
        try:
            response = requests.get(
                f"{API_BASE_URL}/recommend/{article_id}",
                params={"n": top_n}
            )

            if response.status_code == 200:
                data = response.json()

                st.subheader("📄 Reference Article")
                st.write(data["query_title"])

                st.subheader("✨ Recommendations")
                for i, rec in enumerate(data["recommendations"], 1):
                    st.markdown(
                        f"""
                        **{i}. {rec['Title']}**  
                        • ID: `{rec['ID']}`  
                        • Similarity Score: `{rec['Score']:.4f}`
                        """
                    )
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Connection error: {e}")




