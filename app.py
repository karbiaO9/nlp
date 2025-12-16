import streamlit as st
from recommender import recommend

st.set_page_config(page_title="NLP Recommender", page_icon="🧠")
st.title("🧠 Article Recommendation System")

article_id = st.number_input("Article ID", min_value=0, step=1)
top_n = st.slider("Number of recommendations", 1, 10, 3)

if st.button("🔍 Get Recommendations"):
    try:
        results, title = recommend(article_id, top_n)

        st.subheader("📄 Reference Article")
        st.write(title)

        st.subheader("✨ Recommendations")
        for i, r in enumerate(results, 1):
            st.markdown(f"**{i}. {r['Title']}** — score: `{r['Score']:.4f}`")

    except Exception as e:
        st.error(str(e))
