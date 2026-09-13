import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "shap_df" not in st.session_state:
    st.session_state.shap_df = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "xgboost_machine_failure_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error("Unable to load the trained XGBoost model.")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter the machine operating parameters to predict "
    "whether a machine failure may occur."
)


# ============================================================
# INPUT PARAMETERS
# ============================================================

air_temperature = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=298.9,
    step=0.1
)

process_temperature = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=309.1,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational Speed (rpm)",
    min_value=500.0,
    max_value=3000.0,
    value=1500.0,
    step=10.0
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=1.0
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0.0,
    max_value=300.0,
    value=20.0,
    step=1.0
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

temperature_difference = (
    process_temperature - air_temperature
)

load_speed_index = (
    torque * rotational_speed
)


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame(
    {
        "Air_Temperature_K": [air_temperature],
        "Process_Temperature_K": [process_temperature],
        "Rotational_Speed_rpm": [rotational_speed],
        "Torque_Nm": [torque],
        "Tool_Wear_min": [tool_wear],
        "Temperature_Difference_K": [temperature_difference],
        "Load_Speed_Index": [load_speed_index]
    }
)


# ============================================================
# AI EXPLANATION FUNCTION
# ============================================================

def generate_ai_explanation(
    prediction,
    probability,
    input_data,
    shap_df
):

    machine_status = (
        "Machine Failure Detected"
        if prediction == 1
        else "No Machine Failure"
    )

    machine_parameters = input_data.to_dict(
        orient="records"
    )[0]


    # --------------------------------------------------------
    # Identify risk and protective factors deterministically
    # --------------------------------------------------------

    risk_factors = shap_df[
        shap_df["SHAP Value"] > 0
    ]

    protective_factors = shap_df[
        shap_df["SHAP Value"] < 0
    ]


    if len(risk_factors) > 0:

        risk_text = "\n".join(
            [
                f"- {row['Feature']}: "
                f"SHAP = {row['SHAP Value']:.4f}"
                for _, row in risk_factors.iterrows()
            ]
        )

    else:

        risk_text = "- No positive SHAP contributions."


    if len(protective_factors) > 0:

        protective_text = "\n".join(
            [
                f"- {row['Feature']}: "
                f"SHAP = {row['SHAP Value']:.4f}"
                for _, row in protective_factors.iterrows()
            ]
        )

    else:

        protective_text = "- No negative SHAP contributions."


    # --------------------------------------------------------
    # All SHAP contributions
    # --------------------------------------------------------

    shap_text = "\n".join(
        [
            f"- {row['Feature']}: "
            f"SHAP value = {row['SHAP Value']:.4f}"
            for _, row in shap_df.iterrows()
        ]
    )


    # ========================================================
    # PROMPT FOR GEMINI
    # ========================================================

    prompt = f"""
You are an AI-based industrial predictive maintenance assistant.

Analyze the machine prediction using ONLY the information
provided below.

Machine Status:
{machine_status}

Failure Probability:
{probability * 100:.2f}%

Machine Parameters:
{machine_parameters}

All SHAP Feature Contributions:
{shap_text}

Current Risk Factors:
{risk_text}

Current Protective Factors:
{protective_text}


IMPORTANT SHAP RULES:

1. A POSITIVE SHAP value means the CURRENT value of that
feature pushes this specific prediction toward Machine Failure.

2. A NEGATIVE SHAP value means the CURRENT value of that
feature pushes this specific prediction toward No Machine Failure.

3. SHAP values describe contribution to THIS prediction.
They do NOT automatically mean that increasing the feature
will increase or decrease failure probability.

4. Do NOT infer a causal relationship from SHAP values.

5. Do NOT claim that increasing or decreasing a parameter
will definitely increase or decrease failure risk unless
that relationship is explicitly provided.

6. Do NOT invent sensor readings, thresholds, mechanical
faults, measurements, or machine conditions.

7. Only use features with POSITIVE SHAP values as current
risk factors.

8. Only use features with NEGATIVE SHAP values as current
protective factors.

9. The XGBoost model performs the failure prediction.
SHAP provides explainability.
You are only generating a human-readable explanation
and maintenance recommendation.


Generate the response with exactly these sections:

### 1. Explanation

Explain why the XGBoost model produced the current prediction.
Mention the most important SHAP contributions.

### 2. Risk Factors

Mention the current features with positive SHAP values.

### 3. Maintenance Recommendation

Give practical preventive maintenance recommendations
based ONLY on the provided machine parameters and SHAP
contributions.

Do not claim that a specific mechanical fault has been
detected.

Keep the response concise and suitable for an industrial
monitoring application.
"""


    # ========================================================
    # GEMINI API CALL
    # ========================================================

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "AI explanation could not be generated.\n\n"
            "Please check the Gemini API configuration."
        )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Machine Failure Prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]


        # Save prediction in session state
        st.session_state.prediction = prediction
        st.session_state.probability = probability


        # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================

        try:

            explainer = shap.TreeExplainer(
                model
            )

            shap_values = explainer.shap_values(
                input_data
            )


            # Handle different SHAP output formats
            if isinstance(shap_values, list):

                shap_values = shap_values[-1]

            shap_values = np.asarray(
                shap_values
            )

            if shap_values.ndim == 2:

                shap_values = shap_values[0]


            shap_df = pd.DataFrame(
                {
                    "Feature": input_data.columns,
                    "SHAP Value": shap_values
                }
            )


            # Sort by absolute SHAP contribution
            shap_df["Absolute SHAP"] = (
                shap_df["SHAP Value"].abs()
            )

            shap_df = shap_df.sort_values(
                by="Absolute SHAP",
                ascending=False
            ).reset_index(drop=True)


            # Save SHAP result in session state
            st.session_state.shap_df = shap_df.copy()


            # =================================================
            # GEMINI AI EXPLANATION
            # =================================================

            with st.spinner(
                "Generating AI explanation..."
            ):

                ai_explanation = (
                    generate_ai_explanation(
                        prediction,
                        probability,
                        input_data,
                        shap_df
                    )
                )


            # Save AI explanation in session state
            st.session_state.ai_explanation = ai_explanation


        except Exception as shap_error:

            st.session_state.shap_df = None
            st.session_state.ai_explanation = None

            st.warning(
                "Prediction completed, but SHAP/AI explanation "
                "could not be generated."
            )


    except Exception as prediction_error:

        st.error(
            "Prediction failed. Please check the model "
            "and input features."
        )


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction
    probability = st.session_state.probability


    # --------------------------------------------------------
    # Display Failure Probability
    # --------------------------------------------------------

    st.subheader("Failure Probability")

    st.write(
        f"### {probability * 100:.2f}%"
    )


    # --------------------------------------------------------
    # Display Prediction
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "⚠️ Machine Failure Detected"
        )

    else:

        st.success(
            "✅ No Machine Failure"
        )


    # ========================================================
    # SHAP TOGGLE
    # ========================================================

    if st.session_state.shap_df is not None:

        show_shap = st.toggle(
            "Show SHAP Feature Contributions"
        )


        if show_shap:

            st.subheader(
                "SHAP Feature Contributions"
            )

            display_shap = st.session_state.shap_df[
                ["Feature", "SHAP Value"]
            ].copy()

            display_shap["SHAP Value"] = (
                display_shap["SHAP Value"]
                .round(4)
            )

            st.dataframe(
                display_shap,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # GEMINI AI EXPLANATION
    # ========================================================

    if st.session_state.ai_explanation is not None:

        st.subheader(
            "🤖 AI Generated Explanation"
        )

        st.markdown(
            st.session_state.ai_explanation
        )