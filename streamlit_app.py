# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# ==========================
# Streamlit page config
# ==========================
st.set_page_config(page_title="C-section Prediction", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts whether a delivery will be Caesarean (1) or Normal (0).")

# ==========================
# Train model (cached) - WITH SMOTE
# ==========================
@st.cache_resource
def train_model():
    # Load data
    df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
    df = df.dropna(subset=["Delivery by caesarean section"])
    
    target_col = "Delivery by caesarean section"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # Check class distribution
    class_counts = y.value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply SMOTE to training data only
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train models on balanced data
    rf = BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        sampling_strategy='all',
        replacement=True
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=8,
        min_samples_split=10,
        subsample=0.8,
        random_state=42
    )
    
    # Train on balanced data
    rf.fit(X_train_balanced, y_train_balanced)
    gb.fit(X_train_balanced, y_train_balanced)
    
    # Aggressive ensemble favoring C-section detection
    class AggressiveEnsemble:
        def __init__(self, models, csection_boost=3.5):
            self.models = models
            self.csection_boost = csection_boost
            
        def predict_proba(self, X):
            probas = np.array([model.predict_proba(X) for model in self.models])
            avg_proba = np.mean(probas, axis=0)
            
            # Aggressively boost C-section probability
            avg_proba[:, 1] = avg_proba[:, 1] * self.csection_boost
            
            # Renormalize
            row_sums = avg_proba.sum(axis=1, keepdims=True)
            avg_proba = avg_proba / row_sums
            
            return avg_proba
        
        def predict(self, X):
            proba = self.predict_proba(X)
            # Lower threshold for C-section prediction
            return (proba[:, 1] >= 0.35).astype(int)
    
    # Create aggressive ensemble with high boost
    boost_factor = min(4.0, imbalance_ratio / 3)
    model = AggressiveEnsemble([rf, gb], csection_boost=boost_factor)
    
    # Evaluate
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    balanced_counts = pd.Series(y_train_balanced).value_counts()
    
    return model, scaler, test_acc, cm, class_counts, balanced_counts, imbalance_ratio

# Train once and cache
with st.spinner("Training model with SMOTE balancing... (happens once)"):
    model, scaler, test_acc, cm, original_counts, balanced_counts, imbalance_ratio = train_model()

# ==========================
# Display model performance
# ==========================
col1, col2 = st.columns(2)
with col1:
    st.success(f"✅ Test Accuracy: {test_acc:.3f}")
with col2:
    st.warning(f"⚖️ Original Imbalance: {imbalance_ratio:.1f}:1")

col3, col4 = st.columns(2)
with col3:
    st.info(f"📊 Original - Normal: {original_counts.get(0, 0)} | C-section: {original_counts.get(1, 0)}")
with col4:
    st.success(f"✨ After SMOTE - Normal: {balanced_counts.get(0, 0)} | C-section: {balanced_counts.get(1, 0)}")

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
# Prediction
# ==========================
input_scaled = scaler.transform(input_df)
pred = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

st.subheader("🧠 Prediction Result")

# Show probabilities
col1, col2 = st.columns(2)
with col1:
    st.metric("Normal Delivery", f"{pred_proba[0]:.1%}")
with col2:
    st.metric("Caesarean Delivery", f"{pred_proba[1]:.1%}")

# Prediction (lowered threshold to 35%)
if pred == 1:
    st.error(f"⚠️ **CAESAREAN DELIVERY LIKELY** (Probability: {pred_proba[1]:.1%})")
else:
    st.success(f"✅ **Normal delivery likely** (Probability: {pred_proba[0]:.1%})")

# Risk level based on probability
if pred_proba[1] > 0.6:
    st.error("🔴 **HIGH RISK for C-section**")
elif pred_proba[1] > 0.35:
    st.warning("🟡 **MODERATE-HIGH RISK for C-section**")
elif pred_proba[1] > 0.20:
    st.warning("🟠 **MODERATE RISK for C-section**")
else:
    st.info("🟢 Low Risk for C-section")

# Debug
with st.expander("🔍 View Input Values"):
    st.dataframe(input_df)
