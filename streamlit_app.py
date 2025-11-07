import streamlit as st
import pandas as pd

st.title('C-section Prediction App')

st.info('This is an app to run machine learning model.')


with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned.csv')
  df

  
  st.write('**X**')
  X_raw = df.drop('delivery', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.delivery
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='', y='Body_Mass_Index', color='delivery')

