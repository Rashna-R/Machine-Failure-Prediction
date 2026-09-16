import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
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
# GROQ AI FUNCTION
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
        # GET GROQ API KEY
        # -------------------------------------------------

        api_key = st.secrets["GROQ_API_KEY"]

        client = Groq(
            api_key=api_key
        )


        # -------------------------------------------------
        # IDENTIFY SHAP RISK / PROTECTIVE FACTORS
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
        # FORMAT SHAP INFORMATION
        # -------------------------------------------------

        if risk_factors:

            risk_text = "\n".join(
                f"- {item}"
                for item in risk_factors
            )

        else:

            risk_text = "- No positive SHAP contributions."


        if protective_factors:

            protective_text = "\n".join(
                f"- {item}"
                for item in protective_factors
            )

        else:

            protective_text = "- No negative SHAP contributions."


        # -------------------------------------------------
        # GENAI PROMPT
        # -------------------------------------------------

        system_prompt = """
You are an AI-based predictive maintenance assistant.

The machine failure prediction has already been performed
by an XGBoost machine learning model.

Your job is NOT to perform a new prediction.

Your job is to explain the existing model prediction and
provide a practical preventive maintenance recommendation.

Important SHAP interpretation:

Positive SHAP value:
The current feature value contributed toward the model's
failure prediction.

Negative SHAP value:
The current feature value contributed toward the model's
no-failure prediction.

Do not interpret SHAP sign as a general physical or causal rule.

Do not invent:
- mechanical faults
- sensor thresholds
- component damage
- unsupported failure causes

Use only the information provided.

Give the response in exactly three sections:

1. Explanation
2. Risk Factors
3. Maintenance Recommendation

Keep the explanation concise, professional and easy to understand.
"""


        user_prompt = f"""
Machine parameters:

Air Temperature: {air_temperature:.2f} K
Process Temperature: {process_temperature:.2f} K
Rotational Speed: {rotational_speed:.2f} rpm
Torque: {torque:.2f} Nm
Tool Wear: {tool_wear:.2f} min

Model prediction:
{prediction}

Failure probability:
{failure_probability:.2f}%

Current positive SHAP contributions:
{risk_text}

Current negative SHAP contributions:
{protective_text}

Explain the model result based on these values.

For Risk Factors, mention only features having positive SHAP values.

For Maintenance Recommendation, give a practical preventive
maintenance recommendation based only on the available information.
"""


        # -------------------------------------------------
        # GROQ CHAT COMPLETION
        # -------------------------------------------------

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.3,

            max_completion_tokens=1200,
            reasoning_effort="low",
            include_reasoning=False
        )


        # -------------------------------------------------
        # GET RESPONSE
        # -------------------------------------------------

        if response.choices:

            answer = response.choices[0].message.content

            if answer:

                return answer, None

        return None, "Groq returned an empty response."


    except Exception as e:

        return None, str(e)


# =========================================================
# MACHINE PARAMETERS
# =========================================================

st.subheader("🔧 Machine Parameters")


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


torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=0.50
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
    # MODEL INPUT
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

        prediction_value = model.predict(
            input_data
        )[0]

        probability = (
            model.predict_proba(input_data)[0][1]
            * 100
        )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.error(f"Prediction Error: {e}")
        st.stop()


    # -----------------------------------------------------
    # PREDICTION LABEL
    # -----------------------------------------------------

    if prediction_value == 1:

        prediction_text = "Machine Failure"

    else:

        prediction_text = "No Machine Failure"


    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_data
        )


        # Handle SHAP output format
        if isinstance(shap_values, list):

            shap_values = shap_values[-1]


        shap_values = np.asarray(
            shap_values
        )


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
    # SAVE RESULTS
    # -----------------------------------------------------

    st.session_state.prediction = prediction_text

    st.session_state.probability = probability

    st.session_state.shap_df = shap_df

    st.session_state.ai_explanation = None

    st.session_state.ai_error = None


    # -----------------------------------------------------
    # GROQ GENAI
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
# PREDICTION RESULT
# =========================================================

if st.session_state.prediction is not None:

    st.divider()

    st.subheader("📊 Prediction Result")


    st.metric(
        "Failure Probability",
        f"{st.session_state.probability:.2f}%"
    )


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


        display_shap = (
            st.session_state.shap_df[
                ["Feature", "SHAP Value"]
            ]
            .copy()
        )


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
            "Groq AI Error"
        )

        st.code(
            st.session_state.ai_error,
            language="text"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AI-Based Machine Failure Prediction & Maintenance Assistant | "
    "Tuned XGBoost + SHAP + Groq GenAI"
)