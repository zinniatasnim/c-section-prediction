import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------------------
# App title and description
# ----------------------------------------
st.title('🤰 Caesarean Section Prediction App')
st.info('This app predicts whether a delivery will be **Caesarean (C-section)** or **Normal** based on maternal and demographic factors.')

# ----------------------------------------
# Load cleaned data
# ----------------------------------------
df = pd.read_csv('https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv')

# Separate features and target
X = df.drop(columns=['Delivery_by_caesarean_section'])
y = df['Delivery_by_caesarean_section']

# Scale data (if not already scaled)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

st.success("✅ Model trained successfully on cleaned dataset!")

# ----------------------------------------
# Sidebar: Input form
# ----------------------------------------
st.sidebar.header('🧮 Input Features')

# You can replace the default values with dataset median/mode if needed
def user_input_features():
    age = st.sidebar.slider("Respondent's current age", 15, 49, 28)
    bmi = st.sidebar.slider("Body Mass Index (BMI)", 12.0, 60.0, 23.5)
    age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 40, 20)
    residence = st.sidebar.selectbox("Type of place of residence", ['Urban', 'Rural'])
    education = st.sidebar.selectbox("Highest educational level", ['No education', 'Primary', 'Secondary', 'Higher'])
    husband_edu = st.sidebar.selectbox("Husband/partner's education level", ['No education', 'Primary', 'Secondary', 'Higher'])
    wealth = st.sidebar.selectbox("Wealth index combined", ['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'])
    antenatal_visits = st.sidebar.slider("Number of antenatal visits", 0, 30, 5)
    total_children = st.sidebar.slider("Total children ever born", 0, 15, 2)
    
    # Convert to DataFrame
    data = {
        "Respondent's current age": age,
        "Body Mass Index": bmi,
        "Age of respondent at 1st birth": age_first_birth,
        "Type of place of residence": residence,
        "Highest educational level": education,
        "Husband/partner's education level": husband_edu,
        "Wealth index combined": wealth,
        "Number of antenatal visits": antenatal_visits,
        "Total children ever born": total_children
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ----------------------------------------
# Combine with full dataset for consistent encoding
# ----------------------------------------
combined = pd.concat([input_df, X], axis=0)

# One-hot encode categorical variables (match training format)
combined = pd.get_dummies(combined, drop_first=True)
input_encoded = combined[:1]

# Ensure columns match
missing_cols = set(X.columns) - set(input_encoded.columns)
for c in missing_cols:
    input_encoded[c] = 0
input_encoded = input_encoded[X.columns]

# Scale input
input_scaled = scaler.transform(input_encoded)

# ----------------------------------------
# Predict
# ----------------------------------------
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# ----------------------------------------
# Results
# ----------------------------------------
st.subheader('📊 Prediction Results')

df_proba = pd.DataFrame(prediction_proba, columns=['Normal Delivery', 'Caesarean'])
st.dataframe(df_proba.style.format("{:.2f}"))

if prediction[0] == 1:
    st.success("🔴 Predicted Outcome: Caesarean Section")
else:
    st.success("🟢 Predicted Outcome: Normal Delivery")

st.caption("Model: Random Forest Classifier | Trained on cleaned DHS dataset")
