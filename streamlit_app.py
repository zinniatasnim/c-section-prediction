import streamlit as st
import pandas as pd

st.title('C-section Prediction App')

st.info('This is an app to run machine learning model.')

df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned.csv")
df
