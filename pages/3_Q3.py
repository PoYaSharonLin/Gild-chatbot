import streamlit as st
import pandas as pd
import altair as alt
from urllib.error import URLError
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd

st.set_page_config(page_title="Question 3", page_icon="📊")

st.markdown("# Question 3")
st.sidebar.header("Question 3")
st.write(
    """Follow the CBOW example , 
    modify 2 the params, e.g. windows size, 
    vector size, using the new 17 sentences model to try a new sentence."""
)


# Sample sentences
sentences = [
    "A special subset W of a vector space V where applying the operator T to any vector in W keeps the result in W.",
    "An operator T where the inner product <Tu, v> equals <u, Tv> for all vectors u and v, meaning it behaves symmetrically.",
    "A subset W where applying the operator T to any vector in W produces a vector that’s still in W.",
    "An operator T that works the same whether you apply it or its adjoint T^* first.",
    "An operator T where, for any nonzero vector x, the inner product <Tx, x> is positive, like a 'stretching' effect.",
    "An operator T where, for any vector x, the inner product <Tx, x> is zero or positive, never negative.",
    "A number λ where applying the operator T to some nonzero vector x just scales x by λ, i.e., T(x) = λx.",
    "A nonzero vector x that, when the operator T is applied, only gets scaled by some number λ, i.e., T(x) = λx.",
    "A group of vectors that are all perpendicular to each other and each has a length of 1.",
    "A way to break down a linear transformation T into three parts: two rotation-like matrices (U and V) and a diagonal matrix (Σ) with scaling factors called singular values.",
    "Nonnegative numbers in the SVD that show how much a transformation stretches or shrinks vectors.",
    "A square matrix U that preserves lengths and angles, satisfying U^*U = UU^* = I (like a rotation or reflection).",
    "A real square matrix U where U^T U = I, meaning it preserves lengths and angles (a special case of unitary for real numbers).",
    "A way to write a matrix A as A = WP, where W is a rotation-like matrix and P is a matrix that stretches but doesn’t rotate.",
    "A technique to simplify data by fitting it to an ellipsoid and keeping only the directions with the most spread.",
    "A direction w that captures the most variation when data is projected onto it.",
    "A measure of how spread out the data is along a principal component, with PCA focusing on the directions with the most spread."
]

# Preprocess the sentences (remove stopwords)
tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences]

# Define the three sets of parameters
param_sets = [
    {"vector_size": 100, "window": 5, "min_count": 1, "workers": 4, "sg": 0, "name": "Model 1 (vector_size=100, window=5)"},
    {"vector_size": 200, "window": 5, "min_count": 1, "workers": 4, "sg": 0, "name": "Model 2 (vector_size=200, window=5)"},
    {"vector_size": 100, "window": 10, "min_count": 1, "workers": 4, "sg": 0, "name": "Model 3 (vector_size=100, window=10)"}
]

# Train models and get similar words
similar_words_list = []
for params in param_sets:
    model = Word2Vec(
        tokenized_sentences,
        vector_size=params["vector_size"],
        window=params["window"],
        min_count=params["min_count"],
        workers=params["workers"],
        sg=params["sg"]
    )
    # Get most similar words to 'pca'
    try:
        similar_words = model.wv.most_similar('pca', topn=5)  # Top 5 similar words
        similar_words_list.append([(word, round(score, 3)) for word, score in similar_words])
    except KeyError:
        similar_words_list.append([("N/A", 0.0)] * 5)  # Handle case where 'pca' is not in vocabulary

# Streamlit UI
st.title("CBOW Model Comparison: Similarity Rankings for 'pca'")

# Display similar words in a side-by-side table
st.header("Similarity Rankings Comparison")
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

for i, (col, params, similar_words) in enumerate(zip(columns, param_sets, similar_words_list)):
    with col:
        st.subheader(params["name"])
        df = pd.DataFrame(similar_words, columns=["Word", "Similarity Score"])
        st.table(df)