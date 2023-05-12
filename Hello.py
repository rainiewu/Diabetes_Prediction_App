import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# What are my chances of getting diabetes if my relative has it?")

st.markdown(
    """
    some intro here
    ### The tabs on the left:
    - **The Data** comprises of the dataset we use, variables in the model, and assumptions for the model.
    - **The Model** presents our final model and each coefficients.
    - **Play With it** provides the probability of having diabetes after you put in your lifestyle factors.
    ### Want to learn more?
    - Check out our raw code [github.com](website address)
    - Look at the video [presentation](google drive address)
    """
)

  
    
