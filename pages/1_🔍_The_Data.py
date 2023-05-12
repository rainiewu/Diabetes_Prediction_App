import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan



st.set_page_config(page_title="The Data",page_icon="🔍")

st.markdown("# The Data")
st.sidebar.header("The Data")
st.write(
"""Our data is from the 2017-2018 National Health and Nutrition Examination Survey (NHANES) and as they have not distinguished between different types of diabetes, we have combined the different types as one outcome to make the prediction model. Here, we mainly look at lifestyle factors related to diabetes."""
)


# load dataset 
dat = pd.read_csv("diabetesdata2.csv")
diabetes = dat[['diabetes_relative', 'sex', 'ridreth3', 'dmdeduc2', 'dmdmartz', 'ridageyr', 'bmxbmi', 'diabetes']]
diabetes = diabetes.dropna()
# rename dataset
new_column_names = {'diabetes_relative':'Diabetes Relative',
                    'sex':'Sex',
                    'excer':'Exercise',
                    'ridreth3':'Race & Ethnicity', 
                    'dmdeduc2':'Education Level', 
                    'dmdmartz':'Martial Status', 
                    'ridageyr':'Age', 
                    'bmxbmi':'BMI',
                    'diabetes':'Diabetes Status'}
diabetes = diabetes.rename(columns=new_column_names)

# show the dataset
st.dataframe(diabetes)  

st.write("Since we are using demographic data as covariates, we would like to know whether each of the predictors we choose makes some unique contribution in explaining the outcome. Therefore, we make a heatmap to see the correlations between our predictors:")

fig = sns.heatmap(diabetes.corr())
st.pyplot(fig.figure)



