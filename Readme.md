# Diabetes Prediction App

## 1. Description:
Thank you for checking out our App! 

Diabetes affects how our body turns food into energy through insufficient production of insulin, leading to high blood sugar. If left untreated, it can lead to kidney disease, heart disease, and vision loss overtime. It is also the eighth leading cause of death in the US, affecting about 11.3% of the US population with about 28.7 million people diagnosed and 8.5 million undiagnosed. Cases in adults have more than doubled in the last 20 years. 

An accurate prediction of individal's probability of developing diabetes is important for everyone who are considered at risk. Our App uses data from the 2017 to 2018 NHANES (National Health and Nutrition Examination Survey) database to predict the likelihood of an individual developing diabetes. A multivariable logistic model is developed for prediction. Users can input their information into the app, and the model will use this data to estimate their probability of developing diabetes. 

Since our data have not distinguished between different types of diabetes, we have combined the different types as one outcome. By including different crucial variables obtained from the our trained model, our App is able to predict the probability of developing diabetes with over 85% accuracy. 

+ **Instruction**: Our App has a total of 3 sections. 
    1. The Data - This section displays the database we used to train our model. Appropriate assumption checkings are shown as well.
    2. The Model - This section displays the logistic regresion prediction model and its results. 
    3. Play With it - This section is an interface which the users can play around. The users can input values, such as age, or select the options which they think best describe themselves. After the users hit the submit button, an approximated probability of developing diabetes will be shown. 

+ **Prediction Model**: A multivaraible logistic regression model is used to develop the prediction model. Data is divided into training and testing sets. Accuracy, sensitivity, and specificity are calculated and shown. The primary exposure used in the model is whether the individual has a diabetic relative or not. Other covariates such as age, sex, education level, race & ethnicity, exercise level, marital status, and BMI are also considered. 

## 2. Packages:

Our app runs on Streamlit using the programming language Python, version 3.11.2. 

+ import streamlit as st
+ import numpy as np
+ import pandas as pd
+ import seaborn as sns
+ import plotly.express as px
+ import matplotlib.pyplot as plt
+ import statsmodels.formula.api as smf 
+ import plotly.graph_objects as go
+ from sklearn.model_selection import train_test_split
+ from sklearn.linear_model import LogisticRegression
+ from sklearn.metrics import roc_curve


## 3. Division of work:

+ Amy Chen worked on downloading, merging, and cleaning of data, as well as preliminary screening of variables.
+ Rainie Wu worked on development of prediction model using training and validation data.
+ Yingying Yu worked on putting up results together and development of webpages.
+ All members worked on writing descriptions and instructions for the App. 

## 4. Links:

+ Link to App: 

+ Link to video: 

+ Reference: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020

