import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Play With It")

st.markdown("# Play With It")
st.sidebar.header("Play With It")
st.write(
"""Description of how to play with the description model"""
)

with st.form('user_inputs'):  
    col1, col2, col3 = st.columns(3)
    with col1:
        relative = st.radio("Close relative had diabetes?", options=["Yes", "No"],)
    with col2:
        bsex = st.radio("What is your borned sex?", options=["Male", "Female"],)   
    with col3:
        mar = st.radio("What is your marital status?", options=["Married/living with partner", "Widowed/divorced/separated", "Never married"],)

    col4, col5, col6 = st.columns(3)
    with col4:
        age = st.number_input('What is your age?')
    with col5:
        height = st.number_input('What is your height? (inches)')  
    with col6:
        weight = st.number_input('What is your weight? (pounds)')
   
    col7, col8 = st.columns(2)
    with col7:
        race_e = st.radio("What is your race or ethnicity?", options=["Mexican American", "Other Hispanic", "Non-hispanic White", "Non-hispanic Black", "Non-hispanic Asian", "Others/Multi-racial"],)    
    with col8:
        edu = st.radio("What is your education level?", options=["<9th grade", "9-11th grade", "High school grad/GED", "Some college or AA degree", "College graduate or above"],)
    st.form_submit_button()    
    
# transfer inputs into model-readable numbers  
rel, sex, marriage, race, education, bmi = 0, 0, 0, 0, 0, 0
if relative == 'Yes':     
    rel = 0
elif relative == 'No':     
    rel = 1 

if bsex == 'Female':     
    sex = 1 
elif bsex == 'Male':     
    sex = 0

if mar == 'Married/living with partner':     
    marriage = 0
elif mar == 'Widowed/divorced/separated':     
    marriage = 1
elif mar == 'Never married':
    marriage = 2
    
if race_e == 'Mexican American':     
    race = 0
elif race_e == 'Other Hispanic':     
    race = 1
elif race_e == 'Non-hispanic White':
    race = 2
elif race_e == 'Non-hispanic Black':
    race = 3    
elif race_e == 'Non-hispanic Asian':
    race = 4
elif race_e == 'Others/Multi-racial':
    race = 5
    
if edu == '<9th grade':     
    education = 0
elif edu == '9-11th grade':     
    education = 1
elif edu == 'High school grad/GED':
    education = 2
elif edu == 'Some college or AA degree':
    education = 3    
elif edu == 'College graduate or above':
    education = 4
    
bmi = weight / height**2

#st.write('We predict your probability of having diabetes is:'.format(prediction))
st.write('We predict your probability of having diabetes is:')