import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

st.set_page_config(page_title="The Data",page_icon="🔍")

st.markdown("# The Data")
st.sidebar.header("The Data")
st.write(
"""Description of the dataset, variables, assumptions"""
)


# load dataset 
dat = pd.read_csv("https://raw.githubusercontent.com/ds4ph-bme/capstone-project-yyingying00/main/diabetesdata2.csv?token=GHSAT0AAAAAAB57EBEKJ7UQDI4R5SYZ3RHIZC5IY7Q")
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

# checking model assumptions with graphs
#fig1 = px.bar(dat, x = "UN Region", y = "Estimate", color = "Country/Territory")
#fig2 = px.bar(dat, x = "UN Region", y = "Estimate.1", color = "Country/Territory").update_layout(yaxis_title = "Estimate")
#fig3 = px.bar(dat, x = "UN Region", y = "Estimate.2", color = "Country/Territory").update_layout(yaxis_title = "Estimate")
fig4 = sns.heatmap(diabetes.corr())

#st.write("### Checking Model Assumptions:"
#tab1, tab2, tab3, tab4 = st.tabs(["Linearity", "Normality", "Homoscedasticity", "Independence"])
#with tab1:
#    st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
#with tab2:
#    st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
#with tab3:
#    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
#with tab4:
#    st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
  