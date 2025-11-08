# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(page_title="C-section Prediction", layout="centered")
st.title("🤰 Caesarean Section Prediction App")
st.info("Predicts the likelihood of a Caesarean (C-section) delivery based on maternal and socio-economic factors.")

# ===========================
# Load and clean dataset
# ===========================
df = pd.read_csv("clean_csection_data.csv")

df.columns = [
    "Delivery_by_caesarean_section",
    "Current_age",
    "Body_Mass_Index",
    "Age_at_first_birth",
    "Residence_type",
    "Education_level",
    "Husband_education_level",
    "Wealth_index",
    "Antenatal_visits",
    "Total_children_ever_born"
]

# Map categorical to numeric codes
res_map = {"Rural": 0, "Urban": 1}
edu_map = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
wealth_map = {"Poorest": 1, "Poorer": 2, "Middle": 3, "Richer": 4, "Richest": 5}

df["Residence_type"] = df["Residence_type"].map(res_map)
df["Education_level"] = df["Education_level"].map(edu_map)
df["Husband_education_level"] = df["Husband_education_level"].map(edu_map)
df["Wealth_index"] = df["Wealth_index"].map(wealth_map)

# Drop rows with unmapped categories
df = df.dropna().reset_index(drop=True)

# ===========================
# Split X and y
# ===========================
X = df.drop(columns=["Delivery_by_caesarean_section"])
y = df["Delivery_by_caesarean_section"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# ===========================
# Handle imbalance (SMOTE)
# ===========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ===========================
# Random Forest tuning
# ===========================
st.write("🎯 Running Random Forest hyperparameter tuning...")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_distributions=param_grid,
    n_iter=5,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)
rf_search.fit(X_train_res, y_train_res)
best_rf = rf_search.best_estimator_

st.success(f"Best RandomForest Parameters: {rf_search.best_params_}")

# ===========================
# Build stacking model
# ===========================
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
ada = AdaBoostClassifier(n_estimators=150, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

stack_model = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('ada', ada),
        ('xgb', xgb),
        ('gb', gb)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# Train
stack_model.fit(X_train_res, y_train_res)

# ===========================
# Evaluate model
# ===========================
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

# ===========================
# Sidebar Input for Prediction
# ===========================
st.sidebar.header("🔍 Enter Patient Details")

def user_input_features():
    Current_age = st.sidebar.slider("Respondent's current age", 15, 50, 28)
    Body_Mass_Index = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 40.0, 22.0)
    Age_at_first_birth = st.sidebar.slider("Age at 1st birth", 15, 35, 22)
    Residence_type = st.sidebar.selectbox("Residence Type", ('Rural', 'Urban'))
    Education_level = st.sidebar.selectbox("Education Level", ('No education', 'Primary', 'Secondary', 'Higher'))
    Husband_education_level = st.sidebar.selectbox("Husband Education Level", ('No education', 'Primary', 'Secondary', 'Higher'))
    Wealth_index = st.sidebar.selectbox("Wealth Index", ('Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'))
    Antenatal_visits = st.sidebar.slider("Number of Antenatal Visits", 0, 20, 5)
    Total_children_ever_born = st.sidebar.slider("Total children ever born", 0, 10, 2)

    # Apply same encoding as training
    data = {
        'Current_age': Current_age,
        'Body_Mass_Index': Body_Mass_Index,
        'Age_at_first_birth': Age_at_first_birth,
        'Residence_type': res_map[Residence_type],
        'Education_level': edu_map[Education_level],
        'Husband_education_level': edu_map[Husband_education_level],
        'Wealth_index': wealth_map[Wealth_index],
        'Antenatal_visits': Antenatal_visits,
        'Total_children_ever_born': Total_children_ever_born
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ===========================
# Make prediction
# ===========================
input_scaled = scaler.transform(input_df)
prediction = stack_model.predict(input_scaled)[0]
prediction_proba = stack_model.predict_proba(input_scaled)[0]

st.subheader("🧠 Prediction Result")
if prediction == 1:
    st.error("⚠️ Caesarean delivery likely.")
else:
    st.success("✅ Normal delivery likely.")

st.write("### Prediction Probabilities")
st.write(pd.DataFrame([prediction_proba], columns=["Normal", "Caesarean"]))
