# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# ==========================
# Train model (cached) - FIXED VERSION
# ==========================
@st.cache_resource
def train_model():
    # Load data inside the cached function
    df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
    df = df.dropna(subset=["Delivery by caesarean section"])
    
    target_col = "Delivery by caesarean section"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    ada = AdaBoostClassifier(n_estimators=150, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=150, random_state=42)

    stack_model = StackingClassifier(
        estimators=[('rf', rf), ('ada', ada), ('gb', gb)],
        final_estimator=LogisticRegression(class_weight="balanced"),
        cv=5
    )

    stack_model.fit(X_scaled, y)
    
    # Calculate metrics
    cv_scores = cross_val_score(stack_model, X_scaled, y, cv=5, scoring='accuracy')
    cv_acc = cv_scores.mean()
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    y_pred = stack_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return stack_model, scaler, cv_acc, test_acc, cm

# Train once and cache
with st.spinner("Loading model... (this happens only once)"):
    stack_model, scaler, cv_acc, test_acc, cm = train_model()

# ==========================
# Display model performance
# ==========================
st.success(f"📊 CV Accuracy: {cv_acc:.3f}")
st.success(f"✅ Test Accuracy: {test_acc:.3f}")

st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(cm, columns=['Pred_Normal', 'Pred_Caesarean'], index=['Actual_Normal', 'Actual_Caesarean']))

# ==========================
# Sidebar: user input
# ==========================
st.sidebar.header("🔍 Input Patient Data")

age = st.sidebar.slider("Respondent's current age", 15, 49, 28, key="age")
bmi = st.sidebar.slider("Body Mass Index", 10.0, 50.0, 22.5, key="bmi")
age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 40, 20, key="age_first")
antenatal = st.sidebar.slider("Number of antenatal visits", 0, 30, 5, key="antenatal")
total_children = st.sidebar.slider("Total children ever born", 0, 15, 2, key="children")
residence_urban = st.sidebar.selectbox("Type of residence", ["Rural", "Urban"], key="residence")

edu_level = st.sidebar.selectbox("Highest Educational Level", ["No education","Primary","Secondary","Higher"], key="edu")
husband_edu = st.sidebar.selectbox("Husband/Partner's Education Level", ["No education","Primary","Secondary","Higher"], key="husband_edu")
wealth_index = st.sidebar.selectbox("Wealth Index Combined", ["Poorest","Poorer","Middle","Richer","Richest"], key="wealth")

# One-hot encoding
edu_1 = 1 if edu_level == "Primary" else 0
edu_2 = 1 if edu_level == "Secondary" else 0
edu_3 = 1 if edu_level == "Higher" else 0

husband_edu_1 = 1 if husband_edu == "Primary" else 0
husband_edu_2 = 1 if husband_edu == "Secondary" else 0
husband_edu_3 = 1 if husband_edu == "Higher" else 0

wealth_2 = 1 if wealth_index == "Poorer" else 0
wealth_3 = 1 if wealth_index == "Middle" else 0
wealth_4 = 1 if wealth_index == "Richer" else 0
wealth_5 = 1 if wealth_index == "Richest" else 0

input_data = {
    "Respondent's current age": age,
    "Body Mass Index": bmi,
    "Age of respondent at 1st birth": age_first_birth,
    "Number of antenatal visits": antenatal,
    "Total children ever born": total_children,
    "Type of place of residence_2": 1 if residence_urban == "Urban" else 0,
    "Highest educational level_1": edu_1,
    "Highest educational level_2": edu_2,
    "Highest educational level_3": edu_3,
    "Husband/partner's education level_1.0": husband_edu_1,
    "Husband/partner's education level_2.0": husband_edu_2,
    "Husband/partner's education level_3.0": husband_edu_3,
    "Husband/partner's education level_8.0": 0,
    "Wealth index combined_2": wealth_2,
    "Wealth index combined_3": wealth_3,
    "Wealth index combined_4": wealth_4,
    "Wealth index combined_5": wealth_5
}

input_df = pd.DataFrame(input_data, index=[0])

# ==========================
# Prediction (instant!)
# ==========================
input_scaled = scaler.transform(input_df)
pred = stack_model.predict(input_scaled)[0]
pred_proba = stack_model.predict_proba(input_scaled)[0]

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

# Debug: Show input values
with st.expander("🔍 View Input Values (Debug)"):
    st.dataframe(input_df)
