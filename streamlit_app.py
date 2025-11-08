import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Use Altair for plotting (preferred in Streamlit)
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

st.set_page_config(page_title="C-section Prediction (no-mpl)", layout="centered")
st.title("🍼 Caesarean Section Prediction App")
st.info("Predict whether delivery will be Caesarean or Normal using ML. (No matplotlib required)")

# -------------------------
# Load dataset (try local then fallback to GitHub link)
# -------------------------
def load_df():
    # try local file first
    for path in ["cleaned_IR_dataset.csv", "cleaned_for_ml.csv", "cleaned_dataset.csv"]:
        try:
            df_local = pd.read_csv(path)
            st.success(f"Loaded dataset from local file: {path}")
            return df_local
        except Exception:
            pass
    # fallback remote (change this to your repo/file if desired)
    try:
        url = "https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv"
        df_remote = pd.read_csv(url)
        st.success("Loaded dataset from remote GitHub URL")
        return df_remote
    except Exception as e:
        st.error("Could not load dataset from local files or remote URL. Place a CSV named 'cleaned_for_ml.csv' in the app folder or update the URL.")
        st.stop()

df = load_df()

st.write("## Dataset preview")
st.dataframe(df.head())

# Basic column name mapping/fallbacks (try to detect)
possible_target_names = [
    "Delivery_by_caesarean_section",
    "Delivery by caesarean section",
    "delivery_by_caesarean_section",
    "delivery by caesarean section",
    "M17", "m17", "target"
]
target_col = None
for c in possible_target_names:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # fallback: choose a binary-like column
    for col in df.columns:
        uniq = pd.Series(df[col].dropna().unique()).astype(str).str.strip().str.lower()
        if set(uniq).issubset({"0","1","yes","no","true","false","c","cs","caesarean","normal","vaginal"}):
            target_col = col
            break
if target_col is None:
    st.error("Could not detect a target column automatically. Make sure your CSV has a target column indicating c-section (0/1 or yes/no).")
    st.stop()

st.write(f"Using target column: **{target_col}**")

# Common expected column names in your dataset (adjust if your CSV differs)
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

# --- CLEANING ---
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

data = df.copy()
data[target_col] = data[target_col].apply(normalize_target)
data = data.dropna(subset=[target_col]).reset_index(drop=True)
y = data[target_col].astype(int)

# numeric cleaning
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        # replace negative BMI or unrealistic values with median or clip
        if col == "Body Mass Index":
            data[col] = data[col].clip(lower=0)
        data[col] = data[col].fillna(data[col].median())

# categorical cleaning
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        m = data[col].mode(dropna=True)
        if len(m) > 0:
            data[col] = data[col].fillna(m.iloc[0])
        else:
            data[col] = data[col].fillna("missing")

# Prepare unencoded features (for consistent get_dummies)
X_unencoded = data.drop(columns=[target_col]).copy()

# --- ENCODE & TRAIN ---
# one-hot encode categorical columns (safe even if some categories are different)
combined_all = pd.get_dummies(X_unencoded, drop_first=True)
X = combined_all.copy()

# scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# classifier with balanced class weight
clf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

st.success("✅ Model trained (RandomForest with class_weight='balanced')")

# --- EVALUATION ---
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, target_names=["Normal", "C-Section"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

st.write("## Model evaluation")
st.write("### Classification report")
st.dataframe(report_df)

# Confusion matrix plot (Altair if available; otherwise table)
cm_df = pd.DataFrame(cm, index=["Actual_Normal", "Actual_C-Section"], columns=["Pred_Normal", "Pred_C-Section"])
if ALT_AVAILABLE:
    cm_long = (cm_df.reset_index().melt(id_vars="index"))
    cm_long.columns = ["Actual", "Predicted", "Count"]
    chart = alt.Chart(cm_long).mark_rect().encode(
        x=alt.X("Predicted:N", title="Predicted"),
        y=alt.Y("Actual:N", title="Actual"),
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"))
    ).properties(width=400, height=250, title="Confusion Matrix (counts)")
    text = alt.Chart(cm_long).mark_text(baseline="middle", fontSize=12).encode(
        x=alt.X("Predicted:N"),
        y=alt.Y("Actual:N"),
        text=alt.Text("Count:Q")
    )
    st.altair_chart(chart + text, use_container_width=True)
else:
    st.write("Confusion matrix (counts):")
    st.dataframe(cm_df)

# Feature importance (top 10)
feat_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
top10 = feat_importances.head(10).reset_index()
top10.columns = ["feature", "importance"]

st.write("### Top 10 Feature Importances")
if ALT_AVAILABLE:
    bar = alt.Chart(top10).mark_bar().encode(
        x=alt.X("importance:Q"),
        y=alt.Y("feature:N", sort='-x')
    ).properties(width=600, height=300, title="Top 10 Features")
    st.altair_chart(bar, use_container_width=True)
else:
    st.dataframe(top10)

# --- PREDICTION INTERFACE ---
st.sidebar.header("Input for Prediction")

# Provide fixed categorical choices (as you requested)
residence_choice = st.sidebar.selectbox("Type of place of residence", ["Rural", "Urban"])
education_choice = st.sidebar.selectbox("Highest educational level", ["Primary", "High"])
husband_edu_choice = st.sidebar.selectbox("Husband/partner's education level", ["Primary", "High"])
wealth_choice = st.sidebar.selectbox("Wealth index combined", ["Poor", "Middle", "Rich"])

age_val = st.sidebar.slider("Respondent's current age", 10, 60, int(data[numeric_cols[0]].median() if numeric_cols[0] in data else 28))
bmi_val = st.sidebar.slider("Body Mass Index (BMI)", 0.0, 60.0, float(round(data["Body Mass Index"].median() if "Body Mass Index" in data else 22.0,1)))
age1_val = st.sidebar.slider("Age at 1st birth", 10, 50, int(data[numeric_cols[2]].median() if numeric_cols[2] in data else 20))
antenatal_val = st.sidebar.slider("Number of antenatal visits", 0, 30, int(data[numeric_cols[3]].median() if numeric_cols[3] in data else 4))
children_val = st.sidebar.slider("Total children ever born", 0, 20, int(data[numeric_cols[4]].median() if numeric_cols[4] in data else 1))

input_dict = {
    "Respondent's current age": age_val,
    "Body Mass Index": bmi_val,
    "Age of respondent at 1st birth": age1_val,
    "Type of place of residence": residence_choice,
    "Highest educational level": education_choice,
    "Husband/partner's education level": husband_edu_choice,
    "Wealth index combined": wealth_choice,
    "Number of antenatal visits": antenatal_val,
    "Total children ever born": children_val
}
input_df = pd.DataFrame(input_dict, index=[0])
st.write("### Input preview")
st.dataframe(input_df)

# Combine and encode input the same way as training
combined_for_input = pd.concat([input_df, X_unencoded], axis=0, ignore_index=True)
combined_encoded = pd.get_dummies(combined_for_input, drop_first=True)
input_encoded = combined_encoded.iloc[[0]].reindex(columns=X.columns, fill_value=0)

# scale using the same scaler
input_scaled = scaler.transform(input_encoded)

pred = clf.predict(input_scaled)[0]
pred_proba = clf.predict_proba(input_scaled)[0]

st.write("## Prediction")
st.write("**Predicted outcome:**", "🩺 Caesarean (1)" if pred == 1 else "👶 Normal (0)")
st.write("**Probabilities:**")
cols = ["Prob_Normal", "Prob_Caesarean"]
proba_df = pd.DataFrame([pred_proba], columns=cols)
st.dataframe(proba_df.T)

st.caption("If Altair is not installed, plots will be displayed as tables. To enable charts, install Altair (`pip install altair`).")
