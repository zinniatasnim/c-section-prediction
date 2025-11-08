# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================
# Streamlit page config
# ==========================
st.set_page_config(page_title="C-section Prediction", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts whether a delivery will be Caesarean (1) or Normal (0) and shows feature contributions.")

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
# Train model (cached)
# ==========================
@st.cache_resource
def train_model(X, y):
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
    return stack_model, scaler, rf  # Return RF for feature importance

stack_model, scaler, rf_model = train_model(X, y)

# ==========================
# Evaluation
# ==========================
cv_scores = cross_val_score(stack_model, scaler.transform(X), y, cv=5, scoring='accuracy')
cv_acc = cv_scores.mean()
X_train, X_test, y_train, y_test = train_test_split(scaler.transform(X), y, test_size=0.2, stratify=y, random_state=42)
y_pred = stack_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

st.success(f"📊 CV Accuracy: {cv_acc:.3f}")
st.success(f"✅ Test Accuracy: {test_acc:.3f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(pd.DataFrame(cm, columns=['Pred_Normal', 'Pred_Caesarean'], index=['Actual_Normal', 'Actual_Caesarean']))

# ==========================
# Sidebar: user input
# ==========================
st.sidebar.header("🔍 Input Patient Data")

def user_input_features():
    age = st.sidebar.slider("Respondent's current age", 15, 49, 28)
    bmi = st.sidebar.slider("Body Mass Index", 10.0, 50.0, 22.5)
    age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 40, 20)
    antenatal = st.sidebar.slider("Number of antenatal visits", 0, 30, 5)
    total_children = st.sidebar.slider("Total children ever born", 0, 15, 2)
    residence_urban = st.sidebar.selectbox("Type of residence", ["Rural", "Urban"])
    
    edu_level = st.sidebar.selectbox("Highest Educational Level", ["No education","Primary","Secondary","Higher"])
    husband_edu = st.sidebar.selectbox("Husband/Partner's Education Level", ["No education","Primary","Secondary","Higher"])
    wealth_index = st.sidebar.selectbox("Wealth Index Combined", ["Poorest","Poorer","Middle","Richer","Richest"])
    
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

    data = {
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

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==========================
# Prediction
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
st.dataframe(pd.DataFrame([pred_proba], columns=["Normal", "Caesarean"]))

# ==========================
# Feature Contribution / Risk
# ==========================
st.subheader("⚠️ Feature Contribution / Risk Factors")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

# Only show top 5 contributors
top5 = feature_importances_sorted.head(5)
st.write("Top contributing features for prediction:")
for feat, val in top5.items():
    st.write(f"{feat}: {val:.3f}")
