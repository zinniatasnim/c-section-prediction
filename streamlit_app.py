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

st.title("🤖 Caesarean Section Prediction App")
st.info("This AI model predicts the likelihood of a Caesarean (C-section) delivery based on maternal and health factors.")

# ===========================
# Load Data
# ===========================
df = pd.read_csv("clean_csection_data.csv")

# Rename columns (make sure consistent with your CSV)
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

# Separate features and target
X = df.drop(columns=["Delivery_by_caesarean_section"])
y = df["Delivery_by_caesarean_section"]

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=["Residence_type", "Education_level", "Husband_education_level", "Wealth_index"], drop_first=True)

# Scale numeric columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===========================
# Split Data
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ===========================
# Handle Imbalanced Data (SMOTE)
# ===========================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ===========================
# Hyperparameter Tuning for RandomForest
# ===========================
st.write("🎯 Running hyperparameter tuning for Random Forest...")
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

st.success(f"Best RandomForest Params: {rf_search.best_params_}")

# ===========================
# Build Advanced Stacking Model
# ===========================
xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
ada = AdaBoostClassifier(n_estimators=150, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

stacking_model = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('ada', ada),
        ('xgb', xgb),
        ('gb', gb)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

# ===========================
# Train Model
# ===========================
stacking_model.fit(X_train_res, y_train_res)

# ===========================
# Cross-validation accuracy
# ===========================
cv_scores = cross_val_score(stacking_model, X_scaled, y, cv=5, scoring='accuracy')
st.write(f"📊 **Cross-validation Accuracy:** {cv_scores.mean():.3f}")

# ===========================
# Evaluate on Test Data
# ===========================
y_pred = stacking_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.success(f"✅ Test Accuracy: {acc:.3f}")

st.write("### Classification Report")
st.text(classification_report(y_test, y_pred))

st.write("### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
st.dataframe(pd.DataFrame(cm, columns=['Predicted Normal', 'Predicted Caesarean'], index=['Actual Normal', 'Actual Caesarean']))

# ===========================
# Input section for user test
# ===========================
st.sidebar.header("🔍 Enter Patient Details")

def user_input_features():
    Current_age = st.sidebar.slider('Current Age', 15, 50, 28)
    Body_Mass_Index = st.sidebar.slider('Body Mass Index', 10.0, 40.0, 22.0)
    Age_at_first_birth = st.sidebar.slider('Age at 1st Birth', 15, 35, 22)
    Residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    Education_level = st.sidebar.selectbox('Education Level', ('No education', 'Primary', 'Secondary', 'Higher'))
    Husband_education_level = st.sidebar.selectbox('Husband Education Level', ('No education', 'Primary', 'Secondary', 'Higher'))
    Wealth_index = st.sidebar.selectbox('Wealth Index', ('Poor', 'Middle', 'Rich'))
    Antenatal_visits = st.sidebar.slider('Number of Antenatal Visits', 0, 20, 5)
    Total_children_ever_born = st.sidebar.slider('Total Children Ever Born', 0, 10, 2)
    
    data = {
        'Current_age': Current_age,
        'Body_Mass_Index': Body_Mass_Index,
        'Age_at_first_birth': Age_at_first_birth,
        'Residence_type': Residence_type,
        'Education_level': Education_level,
        'Husband_education_level': Husband_education_level,
        'Wealth_index': Wealth_index,
        'Antenatal_visits': Antenatal_visits,
        'Total_children_ever_born': Total_children_ever_born
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Combine with training data for encoding
combined_df = pd.concat([input_df, df.drop(columns=["Delivery_by_caesarean_section"])], axis=0)
combined_encoded = pd.get_dummies(combined_df, columns=["Residence_type", "Education_level", "Husband_education_level", "Wealth_index"], drop_first=True)

input_scaled = scaler.transform(combined_encoded[:1])

prediction = stacking_model.predict(input_scaled)
prediction_proba = stacking_model.predict_proba(input_scaled)

st.subheader("🧠 Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ Caesarean delivery likely.")
else:
    st.success("✅ Normal delivery likely.")

st.write("### Prediction Probabilities")
st.write(pd.DataFrame(prediction_proba, columns=["Normal", "Caesarean"]))
