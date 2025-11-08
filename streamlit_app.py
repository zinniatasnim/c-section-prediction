# streamlit_app_modified.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==========================
# Streamlit page config
# ==========================
st.set_page_config(page_title="C-section Prediction", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts whether a delivery will be Caesarean (1) or Normal (0).")

# ==========================
# Load dataset
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
    df = df.dropna(subset=["Delivery by caesarean section"])
    return df

df = load_data()
target_col = "Delivery by caesarean section"
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# ==========================
# Train logistic regression (cached)
# ==========================
@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model(X, y)

# ==========================
# Sidebar: user input
# ==========================
st.sidebar.header("🔍 Input Patient Data")

age = st.sidebar.slider("Respondent's current age", 15, 49, 28)
bmi = st.sidebar.slider("Body Mass Index", 10.0, 50.0, 22.5)
age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 40, 20)

# Fixed categorical/binary features (remain same)
residence_urban = 0          # 0 = Rural
edu_level_1 = 0              # Primary
edu_level_2 = 0              # Secondary
edu_level_3 = 0              # Higher
husband_edu_1 = 0
husband_edu_2 = 0
husband_edu_3 = 0
husband_edu_8 = 0
wealth_2 = 0
wealth_3 = 0
wealth_4 = 0
wealth_5 = 0

input_data = {
    "Respondent's current age": age,
    "Body Mass Index": bmi,
    "Age of respondent at 1st birth": age_first_birth,
    "Number of antenatal visits": 5,
    "Total children ever born": 2,
    "Type of place of residence_2": residence_urban,
    "Highest educational level_1": edu_level_1,
    "Highest educational level_2": edu_level_2,
    "Highest educational level_3": edu_level_3,
    "Husband/partner's education level_1.0": husband_edu_1,
    "Husband/partner's education level_2.0": husband_edu_2,
    "Husband/partner's education level_3.0": husband_edu_3,
    "Husband/partner's education level_8.0": husband_edu_8,
    "Wealth index combined_2": wealth_2,
    "Wealth index combined_3": wealth_3,
    "Wealth index combined_4": wealth_4,
    "Wealth index combined_5": wealth_5
}

input_df = pd.DataFrame(input_data, index=[0])
input_scaled = scaler.transform(input_df)

# ==========================
# Prediction
# ==========================
pred = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

st.subheader("🧠 Prediction Result")
if pred == 1:
    st.error("⚠️ Caesarean delivery likely (1)")
else:
    st.success("✅ Normal delivery likely (0)")

st.write("### Prediction Probabilities")
prob_df = pd.DataFrame([pred_proba], columns=["Normal", "Caesarean"])
prob_df['Normal'] = prob_df['Normal'].apply(lambda x: f"{x:.1%}")
prob_df['Caesarean'] = prob_df['Caesarean'].apply(lambda x: f"{x:.1%}")
st.dataframe(prob_df)

# Debug
with st.expander("🔍 View Input Values (Debug)"):
    st.dataframe(input_df)
