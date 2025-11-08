# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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
df = pd.read_csv("cleaned_for_ml.csv")

target_col = "Delivery by caesarean section"
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# ==========================
# Train/Test split & scaling
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ==========================
# Define models with class_weight
# ==========================
rf = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
ada = AdaBoostClassifier(n_estimators=150, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, random_state=42)

stack_model = StackingClassifier(
    estimators=[('rf', rf), ('ada', ada), ('gb', gb)],
    final_estimator=LogisticRegression(class_weight="balanced"),
    cv=5
)

# Train model
stack_model.fit(X_train, y_train)

# ==========================
# Evaluation
# ==========================
cv_scores = cross_val_score(stack_model, X_scaled, y, cv=5, scoring='accuracy')
cv_acc = cv_scores.mean()
y_pred = stack_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

st.success(f"📊 Cross-validation Accuracy: {cv_acc:.3f}")
st.success(f"✅ Test Accuracy: {test_acc:.3f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

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
    edu_1 = st.sidebar.checkbox("Education Level 1 (Primary)")
    edu_2 = st.sidebar.checkbox("Education Level 2 (Secondary)")
    edu_3 = st.sidebar.checkbox("Education Level 3 (Higher)")
    husband_edu_1 = st.sidebar.checkbox("Husband Education 1 (Primary)")
    husband_edu_2 = st.sidebar.checkbox("Husband Education 2 (Secondary)")
    husband_edu_3 = st.sidebar.checkbox("Husband Education 3 (Higher)")
    wealth_2 = st.sidebar.checkbox("Wealth Level 2 (Poorer)")
    wealth_3 = st.sidebar.checkbox("Wealth Level 3 (Middle)")
    wealth_4 = st.sidebar.checkbox("Wealth Level 4 (Richer)")
    wealth_5 = st.sidebar.checkbox("Wealth Level 5 (Richest)")

    data = {
        "Respondent's current age": age,
        "Body Mass Index": bmi,
        "Age of respondent at 1st birth": age_first_birth,
        "Number of antenatal visits": antenatal,
        "Total children ever born": total_children,
        "Type of place of residence_2": 1 if residence_urban == "Urban" else 0,
        "Highest educational level_1": int(edu_1),
        "Highest educational level_2": int(edu_2),
        "Highest educational level_3": int(edu_3),
        "Husband/partner's education level_1.0": int(husband_edu_1),
        "Husband/partner's education level_2.0": int(husband_edu_2),
        "Husband/partner's education level_3.0": int(husband_edu_3),
        "Husband/partner's education level_8.0": 0,
        "Wealth index combined_2": int(wealth_2),
        "Wealth index combined_3": int(wealth_3),
        "Wealth index combined_4": int(wealth_4),
        "Wealth index combined_5": int(wealth_5)
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
