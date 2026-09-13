import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import time
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "shap_df" not in st.session_state:
    st.session_state.shap_df = None

if "ai_explanation" not in st.session_state:
    st.session_state.ai_explanation = None

if "ai_error" not in st.session_state:
    st.session_state.ai_error = None


# =========================================================
# TITLE
# =========================================================

st.title("⚙️ AI-Based Machine Failure Prediction")
st.write(
    "Predict machine failure using a tuned XGBoost model "
    "and generate an AI-based maintenance explanation."
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    with open("xgboost_machine_failure_model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


try:
    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")
    st.error(f"Model Error: {e}")
    st.stop()


# =========================================================
# GEMINI AI FUNCTION
# =========================================================

def generate_ai_explanation(
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
    failure_probability,
    prediction,
    shap_df
):

    try:

        # -------------------------------------------------
        # GET GEMINI API KEY
        # -------------------------------------------------

        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(api_key=api_key)


        # -------------------------------------------------
        # SEPARATE RISK AND PROTECTIVE FACTORS
        # -------------------------------------------------

        risk_factors = []
        protective_factors = []

        for _, row in shap_df.iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP Value"]

            if shap_value > 0:
                risk_factors.append(
                    f"{feature}: {shap_value:.4f}"
                )

            elif shap_value < 0:
                protective_factors.append(
                    f"{feature}: {shap_value:.4f}"
                )


        # -------------------------------------------------
        # CONVERT TO TEXT
        # -------------------------------------------------

        if risk_factors:
            risk_text = "\n".join(
                [f"- {x}" for x in risk_factors]
            )
        else:
            risk_text = "- No positive SHAP contributions."


        if protective_factors:
            protective_text = "\n".join(
                [f"- {x}" for x in protective_factors]
            )
        else:
            protective_text = "- No negative SHAP contributions."


        # -------------------------------------------------
        # GEMINI PROMPT
        # -------------------------------------------------

        prompt = f"""
You are an AI-based predictive maintenance assistant.

The machine failure prediction has already been performed by an
XGBoost machine learning model.

Your task is NOT to perform a new prediction.

Your task is to explain the existing prediction in simple,
professional and technically correct language.

MACHINE INPUTS:
- Air Temperature: {air_temperature:.2f} K
- Process Temperature: {process_temperature:.2f} K
- Rotational Speed: {rotational_speed:.2f} rpm
- Torque: {torque:.2f} Nm
- Tool Wear: {tool_wear:.2f} min

MODEL OUTPUT:
- Prediction: {prediction}
- Failure Probability: {failure_probability:.2f}%

SHAP INTERPRETATION:

Positive SHAP value:
The current value of that feature contributed toward the
model's failure prediction.

Negative SHAP value:
The current value of that feature contributed toward the
model's no-failure prediction.

IMPORTANT:
SHAP values describe the contribution of the current feature
value to this particular model prediction.

Do NOT interpret SHAP sign as a general physical or causal rule.

CURRENT POSITIVE SHAP CONTRIBUTIONS:
{risk_text}

CURRENT NEGATIVE SHAP CONTRIBUTIONS:
{protective_text}

Generate the answer using exactly these three sections:

1. Explanation
Explain why the model produced the current prediction,
using the model probability and the important SHAP contributions.

2. Risk Factors
Mention only the features with positive SHAP contributions
as current model risk factors.
Do not invent mechanical faults or unsupported thresholds.

3. Maintenance Recommendation
Give a practical preventive-maintenance recommendation based
only on the provided machine inputs and model/SHAP information.

Keep the answer concise, professional and easy for a machine
maintenance engineer to understand.

Do not claim that a component is definitely damaged.
Do not invent sensor limits, failure thresholds, or mechanical
faults that are not provided.
"""


        # -------------------------------------------------
        # GEMINI API WITH RETRY
        # -------------------------------------------------

        last_error = None

        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )

                if response.text:

                    return response.text, None

                last_error = "Gemini returned an empty response."

            except Exception as e:

                last_error = str(e)

                if attempt < 2:
                    time.sleep(2)


        # If all 3 attempts fail
        return None, last_error


    except Exception as e:

        return None, str(e)


# =========================================================
# USER INPUT SECTION
# =========================================================

st.subheader("🔧 Machine Parameters")

col1, col2 = st.columns(2)


with col1:

    air_temperature = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=298.90,
        step=0.10
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=309.10,
        step=0.10
    )

    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0.0,
        max_value=5000.0,
        value=1500.0,
        step=10.0
    )


with col2:

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        min_value=0.0,
        max_value=300.0,
        value=20.0,
        step=1.0
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Machine Failure",
    type="primary",
    use_container_width=True
):

    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    temperature_difference = (
        process_temperature - air_temperature
    )

    load_speed_index = (
        torque * rotational_speed
    )


    # -----------------------------------------------------
    # CREATE MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame(
        [[
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
            temperature_difference,
            load_speed_index
        ]],
        columns=[
            "Air_Temperature_K",
            "Process_Temperature_K",
            "Rotational_Speed_rpm",
            "Torque_Nm",
            "Tool_Wear_min",
            "Temperature_Difference_K",
            "Load_Speed_Index"
        ]
    )


    # -----------------------------------------------------
    # XGBOOST PREDICTION
    # -----------------------------------------------------

    try:

        prediction_value = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1] * 100


    except Exception as e:

        st.error("❌ Prediction failed.")
        st.error(f"Prediction Error: {e}")
        st.stop()


    # -----------------------------------------------------
    # TEXT PREDICTION
    # -----------------------------------------------------

    if prediction_value == 1:

        prediction_text = "Machine Failure"

    else:

        prediction_text = "No Machine Failure"


    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_data
        )


        # Handle different SHAP output formats
        if isinstance(shap_values, list):

            shap_values = shap_values[-1]

        shap_values = np.asarray(shap_values)

        if shap_values.ndim == 2:

            shap_values = shap_values[0]

        elif shap_values.ndim > 2:

            shap_values = shap_values.reshape(-1)


        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "SHAP Value": shap_values
        })

        shap_df["Absolute SHAP"] = (
            shap_df["SHAP Value"].abs()
        )

        shap_df = shap_df.sort_values(
            by="Absolute SHAP",
            ascending=False
        ).reset_index(drop=True)


    except Exception as e:

        shap_df = None

        st.warning(
            f"SHAP explanation could not be generated: {e}"
        )


    # -----------------------------------------------------
    # STORE RESULTS IN SESSION STATE
    # -----------------------------------------------------

    st.session_state.prediction = prediction_text

    st.session_state.probability = probability

    st.session_state.shap_df = shap_df

    st.session_state.ai_explanation = None

    st.session_state.ai_error = None


    # -----------------------------------------------------
    # GEMINI AI EXPLANATION
    # -----------------------------------------------------

    if shap_df is not None:

        with st.spinner(
            "🤖 Generating AI maintenance explanation..."
        ):

            ai_result, ai_error = generate_ai_explanation(
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                probability,
                prediction_text,
                shap_df
            )


        if ai_result:

            st.session_state.ai_explanation = ai_result

        else:

            st.session_state.ai_error = ai_error


# =========================================================
# DISPLAY PREDICTION
# =========================================================

if st.session_state.prediction is not None:

    st.divider()

    st.subheader("📊 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Failure Probability",
            f"{st.session_state.probability:.2f}%"
        )


    with result_col2:

        if st.session_state.prediction == "Machine Failure":

            st.error(
                "⚠️ Machine Failure Predicted"
            )

        else:

            st.success(
                "✅ No Machine Failure"
            )


# =========================================================
# SHAP TOGGLE
# =========================================================

if st.session_state.shap_df is not None:

    st.divider()

    show_shap = st.toggle(
        "Show SHAP Feature Contributions",
        value=False
    )


    if show_shap:

        st.subheader(
            "🔎 SHAP Feature Contributions"
        )

        display_shap = st.session_state.shap_df[
            ["Feature", "SHAP Value"]
        ].copy()

        display_shap["SHAP Value"] = (
            display_shap["SHAP Value"].round(4)
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Positive SHAP values contributed toward the "
            "failure prediction, while negative SHAP values "
            "contributed toward the no-failure prediction."
        )


# =========================================================
# AI GENERATED EXPLANATION
# =========================================================

if (
    st.session_state.ai_explanation is not None
    or st.session_state.ai_error is not None
):

    st.divider()

    st.subheader(
        "🤖 AI Generated Explanation"
    )


    if st.session_state.ai_explanation:

        st.markdown(
            st.session_state.ai_explanation
        )


    if st.session_state.ai_error:

        st.error(
            "Gemini API Error:"
        )

        st.code(
            st.session_state.ai_error,
            language="text"
        )

        st.info(
            "The XGBoost prediction and SHAP analysis "
            "are working. Only the Gemini explanation "
            "generation failed."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Machine Failure Prediction & Maintenance Assistant | "
    "Tuned XGBoost + SHAP + Gemini"
)