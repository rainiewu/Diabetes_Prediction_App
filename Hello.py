import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# What are your chances of getting diabetes?")

st.markdown(
    """
    ### Background
    Diabetes affects how our body turns food into energy through insufficient production of insulin, leading to high blood sugar. If left untreated, it can lead to kidney disease, heart disease, and vision loss over time. It is also the eighth leading cause of death in the US, affecting about 11.3% of the US population with about 28.7 million people diagnosed and 8.5 million undiagnosed. Cases in adults have more than doubled in the last 20 years.  
    
    There are three types of diabetes and the risk factors are as follow:

    #### Type 1 diabetes 
    - No known prevention as mainly autoimmune 
    - More at risk with family history 
    - You can get type 1 diabetes at any age, but it usually develops in children, teens, or young adults.

    #### Type 2 diabetes 
    - Have prediabetes.
    - Are overweight.
    - Are 45 years or older.
    - Have a parent, brother, or sister with type 2 diabetes.
    - Are physically active less than 3 times a week.
    - Have ever had gestational diabetes (diabetes during pregnancy) or given birth to a baby who weighed over 9 pounds.
    - Are an African American, Hispanic or Latino, American Indian, or Alaska Native person. Some Pacific Islanders and Asian American people are also at higher risk.
    
    #### Gestational Diabetes
    - Had gestational diabetes during a previous pregnancy.
    - Have given birth to a baby who weighed over 9 pounds.
    - Are overweight.
    - Are more than 25 years old.
    - Have a family history of type 2 diabetes.
    - Have a hormone disorder called polycystic ovary syndrome (PCOS).
    - Are an African American, Hispanic or Latino, American Indian, Alaska Native, Native Hawaiian, or Pacific Islander person.

    ### About this website:
    #### The tabs on the left:
    - **The Data** comprises of the dataset we use, variables in the model, and assumptions for the model.
    - **The Model** presents our final model and each coefficients.
    - **Play With it** provides the probability of having diabetes after you put in your demographic and lifestyle info.
    """
)

  
    
##st.write("##### Reference: https://www.cdc.gov/diabetes/basics/diabetes.html")

st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
# Define the CSS styles
styles = {
    "font-size": "24px"
}

# Write text with the defined styles
st.write("Reference: https://www.cdc.gov/diabetes/basics/diabetes.html", unsafe_allow_html=True, style=styles)