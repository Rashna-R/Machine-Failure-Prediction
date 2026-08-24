# Import required libraries
import streamlit as st
import pickle
import numpy as np


# Load the trained machine failure prediction model
with open("xgboost_machine_failure_model.pkl", "rb") as file:
    model = pickle.load(file)


# App title
st.title("Machine Failure Prediction")

# App description
st.write(
    "Enter machine details to predict whether failure may occur."
)


# Input fields
air_temp = st.number_input(
    "Air Temperature (K)",
    value=298.9
)

process_temp = st.number_input(
    "Process Temperature (K)",
    value=309.1
)

rpm = st.number_input(
    "Rotational Speed (rpm)",
    value=2861.0
)

torque = st.number_input(
    "Torque (Nm)",
    value=4.6
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    value=143.0
)


# Prediction
if st.button("Predict"):

    # Prepare input data
    input_data = np.array([
        [air_temp, process_temp, rpm, torque, tool_wear]
    ])

    # Make prediction using the trained model
    prediction = model.predict(input_data)[0]


    # Display prediction result
    if prediction == 1:

        # Display machine failure warning
        st.error("⚠️ Machine Failure Detected")


        # AI Explanation
        st.subheader("🤖 AI Explanation")

        st.write(
            "The machine is predicted to fail based on the given operating conditions. "
            "Factors such as tool wear, rotational speed, temperature and torque "
            "may contribute to the failure."
        )


        # Recommendation
        st.subheader("🔧 Recommendation")

        st.write(
            "Inspect the machine and tool condition immediately. "
            "Check for excessive tool wear, abnormal temperature, torque and rotational speed. "
            "Preventive maintenance is recommended."
        )


    # If machine failure is not predicted
    else:

        # Display success message
        st.success("✅ No Machine Failure")


        # AI Explanation
        st.subheader("🤖 AI Explanation")

        st.write(
            "The machine is currently predicted to operate without failure "
            "based on the given input conditions."
        )


        # Recommendation
        st.subheader("🔧 Recommendation")

        st.write(
            "Continue monitoring the machine parameters and perform regular "
            "preventive maintenance."
        )
