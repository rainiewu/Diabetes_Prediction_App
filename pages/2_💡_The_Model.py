import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, confusion_matrix

st.set_page_config(page_title="The Model")

st.markdown("# The Model")
st.sidebar.header("The Model")
st.write(
"""Description of the model and results"""
)

# load dataset 
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

summary = model_show.summary()

st.text(summary)


# Predict the probability of diabetes for each observation in the test set
y_pred_proba = model.predict_proba(x_test)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Evaluate the performance of the model
st.write('AUC:', roc_auc_score(y_test, y_pred_proba))
st.write('Accuracy:', accuracy_score(y_test, y_pred_proba.round()))
st.write('Specificity:', specificity)
st.write('Sensitivity:', sensitivity)

# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Create Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = %0.2f)' % roc_auc))
fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

# Show Plotly figure in Streamlit
st.plotly_chart(fig)




