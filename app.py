@st.cache_resource
def load_data():
    # Load articles from JSON
    df = pd.read_json("data/articles.json")

    # Keep only required columns
    df = df[["ID", "Title"]]

    # Load embeddings
    embeddings = np.load("data/embeddings_w2v.npy")

    df["_index"] = range(len(df))
    df = df.set_index("ID")

    return df, embeddings
