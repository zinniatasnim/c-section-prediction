import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.title("🤰 Caesarean Section Prediction App")
st.info("This app predicts whether delivery is likely to be by **Caesarean section** using demographic and health data.")

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/zinniatasnim/data/refs/heads/main/cleaned_for_ml.csv")

# Rename columns if needed (ensure names match)
df.columns = [
    "Delivery_by_caesarean_section",
    "Respondent_current_age",
    "Body_Mass_Index",
    "Age_at_first_birth",
    "Place_of_residence",
    "Education_level",
    "Husband_education_level",
    "Wealth_index",
    "Antenatal_visits",
    "Total_children_ever_born"
]

# Show data section
with st.expander("📊 View Dataset"):
    st.write(df.head())

# Split features and target
X = df.drop(columns=["Delivery_by_caesarean_section"])
y = df["Delivery_by_caesarean_section"]

# Encode categorical columns
label_encoders = {}
categorical_cols = ["Place_of_residence", "Education_level", "Husband_education_level", "Wealth_index"]

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Sidebar for input features
st.sidebar.header("🧍 Input Features")

# Get min and max for sliders
age_min, age_max = int(X["Respondent_current_age"].min()), int(X["Respondent_current_age"].max())
bmi_min, bmi_max = int(max(10, X["Body_Mass_Index"].min())), int(X["Body_Mass_Index"].max())  # start from 10 to avoid negatives
age_birth_min, age_birth_max = int(X["Age_at_first_birth"].min()), int(X["Age_at_first_birth"].max())
antenatal_min, antenatal_max = int(X["Antenatal_visits"].min()), int(X["Antenatal_visits"].max())
children_min, children_max = int(X["Total_children_ever_born"].min()), int(X["Total_children_ever_born"].max())

# User inputs
age = st.sidebar.slider("Respondent's current age", age_min, age_max, 25)
bmi = st.sidebar.slider("Body Mass Index", bmi_min, bmi_max, 22)
age_first_birth = st.sidebar.slider("Age at 1st birth", age_birth_min, age_birth_max, 20)
residence = st.sidebar.selectbox("Type of place of residence", X["Place_of_residence"].unique())
education = st.sidebar.selectbox("Highest educational level", X["Education_level"].unique())
husband_edu = st.sidebar.selectbox("Husband/partner's education level", X["Husband_education_level"].unique())
wealth = st.sidebar.selectbox("Wealth index combined", X["Wealth_index"].unique())
antenatal = st.sidebar.slider("Number of antenatal visits", antenatal_min, antenatal_max, 4)
children = st.sidebar.slider("Total children ever born", children_min, children_max, 2)

# Prepare input data
input_data = pd.DataFrame({
    "Respondent_current_age": [age],
    "Body_Mass_Index": [bmi],
    "Age_at_first_birth": [age_first_birth],
    "Place_of_residence": [residence],
    "Education_level": [education],
    "Husband_education_level": [husband_edu],
    "Wealth_index": [wealth],
    "Antenatal_visits": [antenatal],
    "Total_children_ever_born": [children]
})

# Encode input data using same label encoders
for col in categorical_cols:
    input_data[col] = label_encoders[col].transform(input_data[col])

# Train model
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Make prediction
prediction = clf.predict(input_data)
prediction_proba = clf.predict_proba(input_data)

# Display prediction results
st.subheader("🎯 Prediction Result")
if prediction[0] == 1:
    st.success("🩺 **Caesarean section likely**")
else:
    st.info("👶 **Normal delivery likely**")

# Show probability
st.write("**Prediction Probability:**")
st.write(pd.DataFrame(prediction_proba, columns=clf.classes_))
