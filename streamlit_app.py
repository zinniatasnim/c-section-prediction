# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------------------------
# 🎯 App Title and Description
# ------------------------------------
st.title("👶 Caesarean Delivery Prediction App")
st.info("Predict whether a delivery is likely to be **Caesarean** or **Normal**, based on maternal and demographic features.")

# ------------------------------------
# 📂 Load Dataset
# ------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")
    return df

df = load_data()

with st.expander("📊 View Dataset"):
    st.write("**Dataset Sample:**")
    st.dataframe(df.head())
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

# ------------------------------------
# 🧮 Feature & Target Split
# ------------------------------------
X = df.drop("Delivery by caesarean section", axis=1)
y = df["Delivery by caesarean section"]

# ------------------------------------
# ✂️ Train-Test Split
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# 🧠 Model Training
# ------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Model trained successfully! Accuracy on test data: **{acc*100:.2f}%**")

# ------------------------------------
# 🧍‍♀️ User Input Section
# ------------------------------------
st.sidebar.header("Input Mother's Information")

def user_input_features():
    age = st.sidebar.slider("Respondent's current age (scaled)", float(X["Respondent's current age"].min()), float(X["Respondent's current age"].max()), 0.0)
    bmi = st.sidebar.slider("Body Mass Index (scaled)", float(X["Body Mass Index"].min()), float(X["Body Mass Index"].max()), 0.0)
    age_first_birth = st.sidebar.slider("Age of respondent at 1st birth (scaled)", float(X["Age of respondent at 1st birth"].min()), float(X["Age of respondent at 1st birth"].max()), 0.0)
    antenatal_visits = st.sidebar.slider("Number of antenatal visits (scaled)", float(X["Number of antenatal visits"].min()), float(X["Number of antenatal visits"].max()), 0.0)
    total_children = st.sidebar.slider("Total children ever born (scaled)", float(X["Total children ever born"].min()), float(X["Total children ever born"].max()), 0.0)

    # Dynamically include categorical (dummy) columns
    categorical_cols = X.columns[5:]
    categorical_data = {}
    for col in categorical_cols:
        categorical_data[col] = st.sidebar.slider(col, float(X[col].min()), float(X[col].max()), 0.0)

    data = {
        "Respondent's current age": age,
        "Body Mass Index": bmi,
        "Age of respondent at 1st birth": age_first_birth,
        "Number of antenatal visits": antenatal_visits,
        "Total children ever born": total_children,
        **categorical_data
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

with st.expander("👩 User Input Data"):
    st.write(input_df)

# ------------------------------------
# 🔍 Prediction
# ------------------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("🔮 Prediction Result")
st.write(f"**Predicted Delivery Type:** {'🩺 Caesarean' if prediction[0] == 1 else '🤱 Normal'}")

st.subheader("📈 Prediction Probability")
st.dataframe(pd.DataFrame(prediction_proba, columns=['Normal', 'Caesarean']))

# ------------------------------------
# 🎯 Footer
# ------------------------------------
st.caption("Developed with ❤️ using Streamlit and scikit-learn")
