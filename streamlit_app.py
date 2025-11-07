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
        # consider values like '0','1','yes','no','caesarean','normal'
        if set(uniq).issubset({"0", "1", "yes", "no", "true", "false", "caesarean", "c", "normal", "vaginal", "cs"}):
            return col
    # last fallback: pick the last column
    return df.columns[-1]

# -------------------------
# Load original extracted data (unprocessed)
# -------------------------
# Use the extracted 10-column CSV (not the scaled one). This ensures consistent encoding.
try:
    raw = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
except FileNotFoundError:
    st.error("Couldn't find 'new.csv'. Make sure 'new.csv' (the file with your 10 selected columns) is in the app folder.")
    st.stop()

st.write("Preview of loaded columns:")
st.write(raw.columns.tolist())

# Determine target column name
target_col = find_target_column(raw)
st.write(f"Using target column: **{target_col}**")

# -------------------------
# Basic cleaning (same logic as notebook)
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

# Normalize target
data[target_col] = data[target_col].apply(normalize_target)

# Drop rows missing target
data = data.dropna(subset=[target_col]).reset_index(drop=True)

# Identify expected numeric and categorical columns (adjust names if needed)
numeric_cols = [
    "Respondent's current age",
    "Body Mass Index",
    "Age of respondent at 1st birth",
    "Number of antenatal visits",
    "Total children ever born"
]

# infer categorical columns as those that are not numeric and not target
categorical_cols = [c for c in data.columns if c not in numeric_cols + [target_col]]

# Clean numeric: coerce, fill median
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

# Clean categorical: strip, replace 'nan' and fill mode
for col in categorical_cols:
    data[col] = data[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
    mode = data[col].mode(dropna=True)
    if len(mode) > 0:
        data[col] = data[col].fillna(mode[0])
    else:
        data[col] = data[col].fillna("missing")

# -------------------------
# Prepare features and target
# -------------------------
# Keep a copy of original X (unencoded) to allow consistent get_dummies with input row
X_unencoded = data.drop(columns=[target_col]).copy()
y = data[target_col].astype(int)

# -------------------------
# Streamlit: input from user
# -------------------------
st.sidebar.header("Input features")

# Provide defaults using medians/modes from the dataset where possible
def get_mode(col):
    if col in X_unencoded.columns:
        m = X_unencoded[col].mode(dropna=True)
        return m.iloc[0] if len(m) > 0 else ""
    return ""

def get_median(col):
    if col in X_unencoded.columns:
        return float(X_unencoded[col].median())
    return 0.0

age = st.sidebar.slider("Respondent's current age", 10, 60, int(get_median("Respondent's current age")))
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, float(round(get_median("Body Mass Index"),1)))
age_first_birth = st.sidebar.slider("Age at 1st birth", 10, 50, int(get_median("Age of respondent at 1st birth")))
residence = st.sidebar.selectbox("Type of place of residence", sorted(X_unencoded["Type of place of residence"].unique()) if "Type of place of residence" in X_unencoded else [get_mode("Type of place of residence")])
education = st.sidebar.selectbox("Highest educational level", sorted(X_unencoded["Highest educational level"].unique()) if "Highest educational level" in X_unencoded else [get_mode("Highest educational level")])
husband_edu = st.sidebar.selectbox("Husband/partner's education level", sorted(X_unencoded["Husband/partner's education level"].unique()) if "Husband/partner's education level" in X_unencoded else [get_mode("Husband/partner's education level")])
wealth = st.sidebar.selectbox("Wealth index combined", sorted(X_unencoded["Wealth index combined"].unique()) if "Wealth index combined" in X_unencoded else [get_mode("Wealth index combined")])
antenatal_visits = st.sidebar.slider("Number of antenatal visits", 0, 30, int(get_median("Number of antenatal visits")))
total_children = st.sidebar.slider("Total children ever born", 0, 20, int(get_median("Total children ever born")))

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

st.subheader("Input preview")
st.write(input_df)

# -------------------------
# Combine input with dataset for consistent encoding
# -------------------------
combined = pd.concat([input_df, X_unencoded], axis=0, ignore_index=True)

# One-hot encode categorical columns in combined set (this ensures columns match training)
combined_encoded = pd.get_dummies(combined, drop_first=True)

# Take the first row as the encoded input
input_encoded = combined_encoded.iloc[[0]].copy()

# Take the rest as training features and align
X_encoded = combined_encoded.iloc[1:].copy()

# Ensure ordering of columns between X_encoded and input_encoded
for c in X_encoded.columns:
    if c not in input_encoded.columns:
        input_encoded[c] = 0
for c in input_encoded.columns:
    if c not in X_encoded.columns:
        X_encoded[c] = 0

# Reorder columns to be identical
X_encoded = X_encoded.reindex(sorted(X_encoded.columns), axis=1)
input_encoded = input_encoded.reindex(sorted(X_encoded.columns), axis=1)

# -------------------------
# Scale numeric features
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)
input_scaled = scaler.transform(input_encoded)

# -------------------------
# Train a model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

st.success("✅ Model trained successfully (Random Forest)")

# -------------------------
# Predict
# -------------------------
pred = model.predict(input_scaled)[0]
pred_proba = model.predict_proba(input_scaled)[0]

# Map result
label_map = {0: "Normal Delivery", 1: "Caesarean"}
st.subheader("Prediction")
if pred == 1:
    st.error(f"Predicted outcome: {label_map[pred]} (1)")
else:
    st.success(f"Predicted outcome: {label_map[pred]} (0)")

# Show probabilities
proba_df = pd.DataFrame([pred_proba], columns=[f"Prob_{c}" for c in model.classes_])
# Map class numeric labels to readable labels if classes are 0/1
col_names = []
for cl in model.classes_:
    if cl == 0:
        col_names.append("Prob_Normal_Delivery")
    elif cl == 1:
        col_names.append("Prob_Caesarean")
    else:
        col_names.append(f"Prob_{cl}")
proba_df.columns = col_names
st.subheader("Prediction probabilities")
st.write(proba_df.T)

st.caption("Notes: This app recreates the same cleaning/encoding steps on 'new.csv' so user inputs are encoded consistently with training.")
