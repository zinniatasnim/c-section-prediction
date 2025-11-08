import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.title("🍼 Caesarean Section Prediction App")
st.info("Predict whether delivery will be Caesarean or Normal using machine learning.")

# Load dataset
df = pd.read_csv("cleaned_IR_dataset.csv")

st.subheader("📊 Dataset Overview")
st.write(df.head())

# --- Data Preparation ---
# Ensure BMI has no negative values
df["Body_Mass_Index"] = df["Body_Mass_Index"].apply(lambda x: max(x, 0))

# Encode categorical variables
cat_cols = [
    "Type_of_place_of_residence",
    "Highest_educational_level",
    "Husband_partner_education_level",
    "Wealth_index_combined"
]

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Split features and target
X = df_encoded.drop(columns=["Delivery_by_caesarean_section"])
y = df_encoded["Delivery_by_caesarean_section"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# --- Evaluation ---
y_pred = clf.predict(X_test)

st.subheader("📈 Model Performance")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "C-Section"], yticklabels=["Normal", "C-Section"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)

# Classification Report
st.write("### 🧾 Classification Report")
report = classification_report(y_test, y_pred, target_names=["Normal", "C-Section"], output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Feature Importance
st.write("### 🌿 Feature Importance")
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
fig_imp, ax_imp = plt.subplots()
feat_importances.nlargest(10).plot(kind='barh', ax=ax_imp)
plt.title("Top 10 Important Features")
st.pyplot(fig_imp)

# --- Prediction Section ---
st.sidebar.header("🧍 Input for Prediction")

def user_input_features():
    age = st.sidebar.slider("Respondent's Current Age", 15, 49, 28)
    bmi = st.sidebar.slider("Body Mass Index", 0.0, 50.0, 22.5)
    age_first_birth = st.sidebar.slider("Age at First Birth", 12, 40, 20)
    antenatal_visits = st.sidebar.slider("Number of Antenatal Visits", 0, 20, 5)
    total_children = st.sidebar.slider("Total Children Ever Born", 0, 10, 2)
    residence = st.sidebar.selectbox("Type of Residence", df["Type_of_place_of_residence"].unique())
    edu = st.sidebar.selectbox("Highest Education Level", df["Highest_educational_level"].unique())
    husband_edu = st.sidebar.selectbox("Husband/Partner Education Level", df["Husband_partner_education_level"].unique())
    wealth = st.sidebar.selectbox("Wealth Index", df["Wealth_index_combined"].unique())

    data = {
        "Respondents_current_age": age,
        "Body_Mass_Index": bmi,
        "Age_of_respondent_at_1st_birth": age_first_birth,
        "Number_of_antenatal_visits": antenatal_visits,
        "Total_children_ever_born": total_children,
        "Type_of_place_of_residence": residence,
        "Highest_educational_level": edu,
        "Husband_partner_education_level": husband_edu,
        "Wealth_index_combined": wealth
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prepare input for prediction (same encoding)
input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

prediction = clf.predict(input_encoded)[0]
prediction_proba = clf.predict_proba(input_encoded)[0]

st.subheader("🔮 Prediction Result")
st.write("**Predicted Outcome:**", "🩺 Caesarean Section" if prediction == 1 else "👶 Normal Delivery")

st.write("**Prediction Probability:**")
st.write(f"Normal Delivery: {prediction_proba[0]:.2f}")
st.write(f"Caesarean Section: {prediction_proba[1]:.2f}")
