import streamlit as st
import time
import numpy as np

st.set_page_config(page_title=" Question 1 ", page_icon= "📊")

st.markdown(" Question 1 ")
st.sidebar.header(" Question 1 ")
st.write(
    """
•Question 1-1: Select any 10 new sentences and apply it to step 1.1. Provide 2D and 3D views for 10 new sentences. 
•Question 1-2: What is the difference from Word2Vec and SVD. Describe your finding from the result. """
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
