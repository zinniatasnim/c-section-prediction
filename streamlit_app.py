import streamlit as st
import pandas as pd

st.title('C-section Prediction App')

st.info('This is an app to run machine learning model.')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned.csv')
  df

  
  st.write('**x**')
  x_raw = df.drop('delivery', axis=19)
  x_raw

  st.write('**y**')
  y_raw = df.delivery
  y_raw

