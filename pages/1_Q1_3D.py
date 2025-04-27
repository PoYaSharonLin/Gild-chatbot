import streamlit as st
import time
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Question 1-3D", page_icon= "📊")

st.markdown("# Question 1-3D")
st.sidebar.header("Question 1-3D")
st.write(
    """3D view of the 17 new sentences."""
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

# Preprocess the sentences
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the word vectors
word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

# Reduce the dimensions to 3D using PCA
pca = PCA(n_components=3)
reduced_vectors = pca.fit_transform(word_vectors)

# Assign a color to each word based on its sentence
num_sentences = len(sentences)
colors = px.colors.qualitative.Plotly
color_map = {i: colors[i % len(colors)] for i in range(num_sentences)}

# Assign colors to words based on their sentence
word_colors = []
for word in model.wv.index_to_key:
    for i, sentence in enumerate(tokenized_sentences):
        if word in sentence:
            word_colors.append(color_map[i])
            break
    else:
        word_colors.append('gray')

# Create 3D scatter plot for words
scatter = go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers+text',
    text=model.wv.index_to_key,
    textposition='top center',
    marker=dict(color=word_colors, size=2),
    customdata=word_colors,
    hovertemplate="Word: %{text}<br>Color: %{customdata}"
)

# Streamlit UI
st.title("3D Visualization of Word Embeddings")

# Add checkboxes to toggle sentence lines
st.sidebar.header("Select Sentences to Display")
display_array = []
for i in range(num_sentences):
    display_array.append(st.sidebar.checkbox(f"Sentence {i+1}", value=True))

# Create 3D line traces for selected sentences
line_traces = []
for i, sentence in enumerate(tokenized_sentences):
    if display_array[i]:
        line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in sentence if word in model.wv.key_to_index]
        line_trace = go.Scatter3d(
            x=[vector[0] for vector in line_vectors],
            y=[vector[1] for vector in line_vectors],
            z=[vector[2] for vector in line_vectors],
            mode='lines',
            line=dict(color=color_map[i], width=2, dash='solid'),
            showlegend=True,
            name=f"Sentence {i+1}",
            hoverinfo='skip'
        )
        line_traces.append(line_trace)

# Create the figure
fig = go.Figure(data=[scatter] + line_traces)

# Update layout for 3D plot
fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ),
    title="3D Visualization of Word Embeddings",
    width=1000,
    height=1000,
    showlegend=True
)

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
