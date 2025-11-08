import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --- PAGE SETUP ---
st.set_page_config(page_title="C-section Prediction (Stacking Model)", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts whether a delivery will be Caesarean (1) or Normal (0) using a **Stacking ML Model**.")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv"
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

df = load_data()

# --- TARGET COLUMN DETECTION ---
def find_target_column(df):
    candidates = [
        "Delivery_by_caesarean_section",
        "Delivery by caesarean section",
        "delivery_by_caesarean_section",
        "delivery by caesarean section",
        "M17", "m17", "target"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        uniq = pd.Series(df[col].dropna().unique()).astype(str).str.strip().str.lower()
        if set(uniq).issubset({"0","1","yes","no","true","false","normal","vaginal","caesarean","cs"}):
            return col
    return df.columns[-1]

target_col = find_target_column(df)

# --- DATA CLEANING ---
def normalize_target(v):
    if pd.isna(v): return np.nan
    s = str(v).strip().lower()
    if s in {"yes","y","1","true","t","c","c-section","caesarean","cs"}: return 1
    if s in {"no","n","0","false","f","normal","vaginal"}: return 0
    try: return int(float(s))
    except: return np.nan

data = df.copy()
data[target_col] = data[target_col].apply(normalize_target)
data = data.dropna(subset=[target_col]).reset_index(drop=True)

numeric_cols = [
    "Respondent's current age",
    "Body Mass Index",
    "Age of respondent at 1st birth",
    "Number of antenatal visits",
    "Total children ever born"
]
categorical_cols = [
    "Type of place of residence",
    "Highest educational level",
    "Husband/partner's education level",
    "Wealth index combined"
]

# Numeric cleaning
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if col == "Body Mass Index":
            data[col] = data[col].clip(lower=0)
        data[col] = data[col].fillna(data[col].median())

# Categorical cleaning
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        mode = data[col].mode(dropna=True)
        if len(mode) > 0:
            data[col] = data[col].fillna(mode[0])
        else:
            data[col] = data[col].fillna("missing")

# --- FEATURE/TARGET SPLIT ---
X_unencoded = data.drop(columns=[target_col])
y = data[target_col].astype(int)

# --- ENCODING & SCALING ---
X_encoded = pd.get_dummies(X_unencoded, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- STACKING MODEL ---
base_models = [
    ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=-1
)

stacking_model.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.success(f"✅ Model trained successfully using **Stacking Ensemble**")
st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")

st.subheader("📊 Classification Report")
st.dataframe(report_df)

# --- INPUT FORM ---
st.sidebar.header("🔍 Enter Patient Data")

residence = st.sidebar.selectbox("Type of place of residence", ["Rural", "Urban"])
education = st.sidebar.selectbox("Highest educational level", ["Primary", "High"])
husband_edu = st.sidebar.selectbox("Husband/partner's education level", ["Primary", "High"])
wealth = st.sidebar.selectbox("Wealth index combined", ["Poor", "Middle", "Rich"])

age = st.sidebar.slider("Respondent's current age", 10, 60, 25)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 0.0, 60.0, 22.0)
age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 50, 20)
antenatal = st.sidebar.slider("Number of antenatal visits", 0, 30, 4)
children = st.sidebar.slider("Total children ever born", 0, 20, 2)

input_dict = {
    "Respondent's current age": age,
    "Body Mass Index": bmi,
    "Age of respondent at 1st birth": age_first_birth,
    "Type of place of residence": residence,
    "Highest educational level": education,
    "Husband/partner's education level": husband_edu,
    "Wealth index combined": wealth,
    "Number of antenatal visits": antenatal,
    "Total children ever born": children
}
input_df = pd.DataFrame(input_dict, index=[0])

# Combine with training data for consistent encoding
combined = pd.concat([input_df, X_unencoded], axis=0, ignore_index=True)
combined_encoded = pd.get_dummies(combined, drop_first=True)
input_encoded = combined_encoded.iloc[[0]].reindex(columns=X_encoded.columns, fill_value=0)
input_scaled = scaler.transform(input_encoded)

# --- PREDICTION ---
pred = stacking_model.predict(input_scaled)[0]
pred_proba = stacking_model.predict_proba(input_scaled)[0]

st.subheader("🎯 Prediction Result")
if pred == 1:
    st.error("**Predicted Outcome:** Caesarean Delivery (1)")
else:
    st.success("**Predicted Outcome:** Normal Delivery (0)")

proba_df = pd.DataFrame([pred_proba], columns=["Prob_Normal", "Prob_Caesarean"])
st.write("### Probability Scores")
st.dataframe(proba_df.T)

st.caption("Stacking model combines Logistic Regression, KNN, Random Forest, and AdaBoost for higher predictive accuracy.")
