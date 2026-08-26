# MACHINE FAILURE PREDICTION

## Project Description

This project predicts whether a machine failure may occur based on machine operating conditions using a Machine Learning model.

The application is built using Streamlit and provides an interactive interface for entering machine parameters and obtaining a failure prediction.

## Input Features

1. Air Temperature (K)
2. Process Temperature (K)
3. Rotational Speed (rpm)
4. Torque (Nm)
5. Tool Wear (min)

## Machine Learning Model

XGBoost

Model File:
xgboost_machine_failure_model.pkl

## Application

Streamlit

## Explainable AI

The application includes SHAP-based Explainable AI to help understand how the input machine parameters contribute to the prediction.

The user can switch between:

- Without SHAP — Displays the machine failure prediction with AI explanation and recommendation.
- With SHAP — Displays the prediction along with SHAP feature contributions, main contributing factor, risk factors, protective factors, and SHAP summary.

## Application Features

- Machine Failure Prediction
- No Machine Failure Prediction
- AI-based Explanation
- Preventive Maintenance Recommendation
- SHAP Explainability
- Risk Factors
- Protective Factors
- Main Contributing Factor
- Interactive SHAP Toggle

## Output

The application predicts one of the following:

1. Machine Failure Detected
2. No Machine Failure

Based on the prediction, the application provides an AI-based explanation and recommendation.

When SHAP is enabled, the application also provides detailed feature contribution information to explain the prediction.

## Project Components

- Machine_Failure_Analysis.ipynb
- xgboost_machine_failure_model.pkl
- streamlit_app.py
- requirements.txt

## Technologies Used

- Python
- Pandas
- XGBoost
- SHAP
- Streamlit
- Matplotlib

## Live Demo : https://machine-failure-prediction-e5sefslugteb2uh5qxthvr.streamlit.app/



