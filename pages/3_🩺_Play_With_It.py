import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Play With It")

st.markdown("# Play With It")
st.sidebar.header("Play With It")
st.write(
"""
    Answer the survey below by plugging in your info to see what are your chances of getting diabetes!  
    **Don't worry!** Your information would not be saved or shared with anyone.
"""
)

with st.form('user_inputs'):  
    col1, col2, col3 = st.columns(3)
    with col1:
        relative = st.radio("Close relative had diabetes?", options=["Yes", "No"],)
    with col2:
        bsex = st.radio("What is your borned sex?", options=["Male", "Female"],)   
    with col3:
        exer = st.radio("Do you exercise regularly?", options=["Yes", "No"],)

    col4, col5, col6 = st.columns(3)
    with col4:
        age = st.number_input('What is your age?', value = 20)
    with col5:
        height = st.number_input('What is your height? (inches)', value = 70)  
    with col6:
        weight = st.number_input('What is your weight? (pounds)', value = 100)
   
    col7, col8, col9 = st.columns(3)
    with col7:
        race_e = st.radio("What is your race or ethnicity?", options=["Mexican American", "Other Hispanic", "Non-hispanic White", "Non-hispanic Black", "Non-hispanic Asian", "Others/Multi-racial"],)    
    with col8:
        edu = st.radio("What is your education level?", options=["<9th grade", "9-11th grade", "High school grad/GED", "Some college or AA degree", "College graduate or above"],)
    st.form_submit_button()  
    with col9:
        mar = st.radio("What is your marital status?", options=["Married/living with partner", "Widowed/divorced/separated", "Never married"],)
    
# transfer inputs into model-readable numbers  
rel, sex, marriage, race, education, exercise, bmi = 0, 0, 0, 0, 0, 0, 0
if relative == 'Yes':     
    rel = 1
elif relative == 'No':     
    rel = 0

if bsex == 'Female':     
    sex = 1 
elif bsex == 'Male':     
    sex = 0

if exer == 'Yes':     
    exercise = 1
elif relative == 'No':     
    exercise = 0
    
if mar == 'Married/living with partner':     
    marriage = 0
elif mar == 'Widowed/divorced/separated':     
    marriage = 1
elif mar == 'Never married':
    marriage = 2
    
if race_e == 'Mexican American':     
    race = 1
elif race_e == 'Other Hispanic':     
    race = 2
elif race_e == 'Non-hispanic White':
    race = 3
elif race_e == 'Non-hispanic Black':
    race = 4    
elif race_e == 'Non-hispanic Asian':
    race = 6
elif race_e == 'Others/Multi-racial':
    race = 7
    
if edu == '<9th grade':     
    education = 1
elif edu == '9-11th grade':     
    education = 2
elif edu == 'High school grad/GED':
    education = 3
elif edu == 'Some college or AA degree':
    education = 4    
elif edu == 'College graduate or above':
    education = 5
    
bmi = 703 * weight / height**2 

new_data = pd.DataFrame([rel, exercise, sex,race,education,marriage,age,bmi]).T
new_data = sm.add_constant(new_data, has_constant='add')


## load the model

dat = pd.read_csv("diabetesdata2.csv")
diabetes = dat[['diabetes_relative', 'sex', 'ridreth3','excer', 'dmdeduc2', 'dmdmartz', 'ridageyr', 'bmxbmi', 'diabetes']]
diabetes = diabetes.dropna()
# rename dataset
new_column_names = {'diabetes_relative':'diabetes relative',
                    'ridreth3':'race & ethnicity', 
                    'excer':'excercise',
                    'dmdeduc2':'education level', 
                    'dmdmartz':'martial status', 
                    'ridageyr':'age', 
                    'bmxbmi':'bmi'}
diabetes = diabetes.rename(columns=new_column_names)

threshold = 0.2

# Split the dataset into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.drop(columns=['diabetes']), diabetes['diabetes'], test_size=0.15, random_state=63)

# Create the logistic regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training set
model.fit(x_train, y_train)

# Adding a constant term
x_train = sm.add_constant(x_train)
# Fitting the logistic regression model
model_show = sm.Logit(y_train, x_train).fit()

prediction = model_show.predict(new_data)
st.write("Based on our model, we predict your probability of having diabetes is:")
st.markdown(f"<h1 style='font-size: 70px;'>{prediction[0] * 100:.2f}%</h1>", unsafe_allow_html=True)

st.write("(Our model is very simple, based on a few lifestyle factors related to diabetes and not including lab data such as fasting blood glucose level. Please consult a doctor if you think you may be at risk for diabetes.)")
         
    
