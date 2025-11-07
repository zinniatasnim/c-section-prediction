import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="C-section Prediction", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts whether a delivery will be Caesarean (1) or Normal (0).")

# -------------------------
# Helper: find target column
# -------------------------
def find_target_column(df):
    candidates = [
        "Delivery_by_caesarean_section",
        "Delivery by caesarean section",
        "delivery_by_caesarean_section",
        "delivery by caesarean section",
        "m17", "M17", "target"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: try to find a column that is binary (only 0 and 1)
    for col in df.columns:
        uniq = pd.Series(df[col].dropna().unique()).astype(str).str.strip().str.lower()
        if set(uniq).issubset({"0","1","yes","no","true","false","caesarean","normal","vaginal","cs"}):
            return col
    return df.columns[-1]

# -------------------------
# Load dataset
# -------------------------
try:
    raw = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
except Exception as e:
    st.error("⚠️ Could not load 'cleaned_for_ml.csv'. Please check the file path or link.")
    st.stop()

st.write("📄 **Loaded Columns:**")
st.write(raw.columns.tolist())

# Identify target column
target_col = find_target_column(raw)
st.write(f"🎯 Using target column: **{target_col}**")

# -------------------------
# Clean and prepare dataset
# -------------------------
def normalize_target(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in {"yes","y","1","true","t","c","c-section","caesarean","caesarean section","cs"}:
        return 1
    if s in {"no","n","0","false","f","normal","vaginal","vaginal delivery"}:
        return 0
    try:
        return int(float(s))
    except:
        return np.nan

data = raw.copy()
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

# Clean numeric columns
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

# Fill missing categorical with mode
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        mode = data[col].mode(dropna=True)
        data[col] = data[col].fillna(mode[0] if len(mode) > 0 else "missing")

X_unencoded = data.drop(columns=[target_col])
y = data[target_col].astype(int)

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.header("🧍 Input Features")

def get_median(col):
    if col in X_unencoded.columns:
        return float(X_unencoded[col].median())
    return 0.0

age = st.sidebar.slider("Respondent's current age", 10, 60, int(get_median("Respondent's current age")))
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, float(round(get_median("Body Mass Index"),1)))
age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 50, int(get_median("Age of respondent at 1st birth")))

# ✅ Fixed categorical options
residence = st.sidebar.selectbox("Type of place of residence", ["Rural", "Urban"])
education = st.sidebar.selectbox("Highest educational level", ["Primary", "High"])
husband_edu = st.sidebar.selectbox("Husband/partner's education level", ["Primary", "High"])
wealth = st.sidebar.selectbox("Wealth index combined", ["Poor", "Middle", "Rich"])

antenatal_visits = st.sidebar.slider("Number of antenatal visits", 0, 30, int(get_median("Number of antenatal visits")))
total_children = st.sidebar.slider("Total children ever born", 0, 20, int(get_median("Total children ever born")))

# Combine user inputs into a DataFrame
input_data = {
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
input_df = pd.DataFrame(input_data, index=[0])

st.subheader("📋 Input Preview")
st.write(input_df)

# -------------------------
# One-hot encode
# -------------------------
combined = pd.concat([input_df, X_unencoded], axis=0, ignore_index=True)
combined_encoded = pd.get_dummies(combined, drop_first=True)

input_encoded = combined_encoded.iloc[[0]].copy()
X_encoded = combined_encoded.iloc[1:].copy()

# Align columns
for c in X_encoded.columns:
    if c not in input_encoded.columns:
        input_encoded[c] = 0
for c in input_encoded.columns:
    if c not in X_encoded.columns:
        X_encoded[c] = 0

X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)
input_encoded = input_encoded.reindex(sorted(X_encoded.columns), axis=1)

# -------------------------
# Scale numeric features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
input_scaled = scaler.transform(input_encoded)

# -------------------------
# Train model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

st.success("✅ Model trained successfully (Random Forest)")

# -------------------------
# Predict
# -------------------------
pred = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

label_map = {0: "Normal Delivery", 1: "Caesarean"}
st.subheader("🎯 Prediction Result")
if pred == 1:
    st.error(f"Predicted outcome: {label_map[pred]} (1)")
else:
    st.success(f"Predicted outcome: {label_map[pred]} (0)")

proba_df = pd.DataFrame([pred_proba], columns=[f"Prob_{c}" for c in model.classes_])
proba_df.columns = ["Prob_Normal_Delivery", "Prob_Caesarean"]
st.subheader("📈 Prediction Probabilities")
st.write(proba_df.T)

st.caption("🔍 This app reproduces the same data cleaning and encoding steps used for model training.")
