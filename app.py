import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load pre-trained files
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("selected_features.pkl", "rb") as file:
    selected_features = pickle.load(file)

with open("selected_scaler.pkl", "rb") as file:
    selected_scaler = pickle.load(file)

st.title("Obesity Level Prediction Based on Eating Habits and Physical Condition")
st.write("This application uses XGBoost to categorise obesity levels into different classes.")

st.subheader("Enter Your Information")

input_data = {}
if "Gender" in selected_features:
    gender = st.selectbox("Gender", ["Male", "Female"])
    input_data["Gender"] = 1 if gender == 'Male' else 0

if "Age" in selected_features:
    age = st.number_input("Age", min_value=0, max_value=100)
    input_data["Age"] = age

height = 0  
weight = 0 

height = st.number_input("Height (cm)", min_value=50, max_value=250)

if "Weight" in selected_features:
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200)
    input_data["Weight"] = weight

# Calculate BMI dynamically if required
if "BMI" in selected_features:
    if height > 0 and weight > 0:
        bmi = round(weight / ((height / 100) ** 2), 2)
        input_data["BMI"] = bmi
        st.write(f"**Your calculated BMI is:** {bmi}")
    else:
        input_data["BMI"] = None
        st.write("Please enter valid height and weight to calculate BMI.")

if "family_history_with_overweight" in selected_features:
    family_history = st.selectbox("Do you have any family history of being overweight?", ["Yes", "No"])
    input_data["family_history_with_overweight"] = 1 if family_history == "Yes" else 0

if "FAVC" in selected_features:
    favc = st.selectbox("Do you eat high-caloric food frequently?", ["Yes", "No"])
    input_data["FAVC"] = 1 if favc == "Yes" else 0

if "FCVC" in selected_features:
    fcvc = st.number_input("Rate the amount of veggies in your meals (1-3)", min_value=1, max_value=3)
    input_data["FCVC"] = fcvc

if "NCP" in selected_features:
    ncp = st.number_input("Number of main meals per day (1-4)", min_value=1, max_value=4)
    input_data["NCP"] = ncp

if "CAEC" in selected_features:
    caec = st.number_input("Rate snacking between meals (0-3)", min_value=0, max_value=3)
    input_data["CAEC"] = caec

if "SMOKE" in selected_features:
    smoke = st.selectbox("Do you smoke?", ["Yes", "No"])
    input_data["SMOKE"] = 1 if smoke == "Yes" else 0

if "CH2O" in selected_features:
    ch2o = st.number_input("Daily water consumption (1-3 liters)", min_value=1, max_value=3)
    input_data["CH2O"] = ch2o

if "SCC" in selected_features:
    scc = st.selectbox("Do you monitor your daily calorie intake?", ["Yes", "No"])
    input_data["SCC"] = 1 if scc == "Yes" else 0

if "FAF" in selected_features:
    faf = st.selectbox("How often do you exercise?", ["Low", "Moderate", "High"])
    input_data["FAF"] = 2 if faf == "High" else (1 if faf == "Moderate" else 0)

if "TUE" in selected_features:
    tue = st.selectbox("Time spent using electronic devices?", ["Low", "Moderate", "High"])
    input_data["TUE"] = 2 if tue == "High" else (1 if tue == "Moderate" else 0)

if "CALC" in selected_features:
    calc = st.selectbox("How often do you consume alcohol?", ["Never", "Sometimes", "Frequently"])
    input_data["CALC"] = 0 if calc == "Never" else (1 if calc == "Sometimes" else 2)


# Prepare input data for prediction

input_df = pd.DataFrame([input_data])[selected_features]

expected_features = selected_scaler.feature_names_in_ 

for feature in expected_features:
    if feature not in input_df.columns:
        input_df[feature] = 0

input_df = input_df[expected_features] 

input_scaled = selected_scaler.transform(input_df)
input_scaled = np.array(input_scaled) 


if st.button("Predict"):
    prediction = model.predict(input_scaled.reshape(1,-1))
    predicted_label = prediction[0]

    label_mapping = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'}

    result = label_mapping.get(predicted_label, "Unknown")
    st.write(f'The predicted test result is : **{result}**')
else:
    st.write("Please provide height and weight to calculate BMI before predicting.")