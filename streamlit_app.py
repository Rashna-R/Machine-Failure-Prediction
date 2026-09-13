import streamlit as st
import pandas as pd
import pickle
import shap
import ollama


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)


# =====================================================
# LOAD FINAL MODEL
# =====================================================

MODEL_PATH = "xgboost_machine_failure_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# =====================================================
# GENERATE AI EXPLANATION USING LOCAL OLLAMA
# =====================================================

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

    shap_information = []

    for _, row in shap_df.iterrows():

        shap_information.append(
            f"{row['Feature']}: "
            f"SHAP value = {row['SHAP Value']:.4f}"
        )

    shap_text = "\n".join(shap_information)


    # -------------------------------------------------
    # LLM Prompt
    # -------------------------------------------------

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

SHAP Feature Contributions:
{shap_text}


SHAP Interpretation Rules:

1. A positive SHAP value means that the feature's current
   value pushes the model prediction toward Machine Failure.

2. A negative SHAP value means that the feature's current
   value pushes the model prediction toward No Machine Failure.

3. SHAP values describe the contribution of the CURRENT
   feature value to this specific prediction.

4. Do NOT claim that increasing a feature will necessarily
   increase or decrease failure probability unless this
   relationship is directly supported by the provided data.

5. Do NOT interpret a negative SHAP value as evidence that
   a higher feature value causes higher failure risk.

6. Do not invent sensor readings, failure causes, thresholds,
   measurements, or machine conditions that are not provided.

7. When discussing risk factors, identify features with
   positive SHAP values.

8. When discussing protective factors, identify features with
   negative SHAP values.


Generate a concise response with these sections:

1. Explanation

Explain why the model produced the current prediction,
using the most important SHAP feature contributions.

2. Risk Factors

Mention the features with positive SHAP values that are
currently pushing the prediction toward Machine Failure.

3. Maintenance Recommendation

Give practical preventive maintenance recommendations
based ONLY on the available machine parameters and SHAP
contributions.

Do not claim that the model has detected a specific
mechanical fault unless such information is explicitly
provided.

Keep the response suitable for an industrial monitoring
application.
"""


    # -------------------------------------------------
    # Local Ollama LLM
    # -------------------------------------------------

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =====================================================
# APPLICATION TITLE
# =====================================================

st.title("⚙️ Machine Failure Prediction")

st.write(
    "Enter machine operating parameters to predict "
    "whether machine failure may occur."
)


# =====================================================
# INPUT PARAMETERS
# =====================================================

air_temperature = st.number_input(
    "Air Temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=298.9,
    step=0.1
)

process_temperature = st.number_input(
    "Process Temperature (K)",
    min_value=250.0,
    max_value=450.0,
    value=309.1,
    step=0.1
)

rotational_speed = st.number_input(
    "Rotational Speed (rpm)",
    min_value=500.0,
    max_value=5000.0,
    value=1500.0,
    step=1.0
)

torque = st.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=30.0,
    step=0.1
)

tool_wear = st.number_input(
    "Tool Wear (min)",
    min_value=0.0,
    max_value=300.0,
    value=20.0,
    step=1.0
)


# =====================================================
# SHAP OPTION
# =====================================================

show_shap = st.toggle(
    "🔍 Show SHAP Explanation",
    value=False
)


# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict"):

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------

    temperature_difference = (
        process_temperature - air_temperature
    )

    load_speed_index = (
        torque * rotational_speed
    )


    # -------------------------------------------------
    # Prepare Input Data
    # -------------------------------------------------

    input_data = pd.DataFrame({

        "Air_Temperature_K": [
            air_temperature
        ],

        "Process_Temperature_K": [
            process_temperature
        ],

        "Rotational_Speed_rpm": [
            rotational_speed
        ],

        "Torque_Nm": [
            torque
        ],

        "Tool_Wear_min": [
            tool_wear
        ],

        "Temperature_Difference_K": [
            temperature_difference
        ],

        "Load_Speed_Index": [
            load_speed_index
        ]
    })


    # -------------------------------------------------
    # Model Feature Order
    # -------------------------------------------------

    model_features = [

        "Air_Temperature_K",

        "Process_Temperature_K",

        "Rotational_Speed_rpm",

        "Torque_Nm",

        "Tool_Wear_min",

        "Temperature_Difference_K",

        "Load_Speed_Index"
    ]

    input_data = input_data[
        model_features
    ]


    # -------------------------------------------------
    # Model Prediction
    # -------------------------------------------------

    prediction = int(
        model.predict(input_data)[0]
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )


    # -------------------------------------------------
    # SHAP Analysis
    # -------------------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data
    )


    if isinstance(shap_values, list):

        shap_values = shap_values[-1]


    if len(shap_values.shape) > 1:

        shap_values = shap_values[0]


    shap_df = pd.DataFrame({

        "Feature": input_data.columns,

        "SHAP Value": shap_values
    })


    shap_df["Impact"] = (
        shap_df["SHAP Value"].abs()
    )


    shap_df = shap_df.sort_values(
        "Impact",
        ascending=False
    ).reset_index(drop=True)


    # =================================================
    # FAILURE PROBABILITY
    # =================================================

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )


    # =================================================
    # PREDICTION RESULT
    # =================================================

    if prediction == 1:

        st.error(
            "⚠️ Machine Failure Detected"
        )

    else:

        st.success(
            "✅ No Machine Failure"
        )


    # =================================================
    # AI GENERATED EXPLANATION
    # =================================================

    st.subheader(
        "🤖 AI Generated Explanation"
    )


    try:

        ai_response = generate_ai_explanation(
            prediction,
            probability,
            input_data,
            shap_df
        )

        st.write(ai_response)


    except Exception as e:

        st.warning(
            "AI explanation could not be generated."
        )

        st.code(str(e))


    # =================================================
    # SHAP EXPLANATION
    # =================================================

    if show_shap:

        st.subheader(
            "📊 SHAP Feature Contribution"
        )


        # -------------------------------------------------
        # Main Contributing Feature
        # -------------------------------------------------

        main_feature = (
            shap_df.iloc[0]["Feature"]
        )

        main_value = (
            shap_df.iloc[0]["SHAP Value"]
        )


        st.info(
            f"🎯 Main Contributing Factor: "
            f"{main_feature}"
        )


        # -------------------------------------------------
        # SHAP Table
        # -------------------------------------------------

        display_df = shap_df[
            ["Feature", "SHAP Value"]
        ].copy()


        display_df["Effect"] = display_df[
            "SHAP Value"
        ].apply(

            lambda value:

            "🔴 Increases Failure Risk"

            if value > 0

            else

            "🟢 Reduces Failure Risk"

            if value < 0

            else

            "⚪ Very Low Impact"
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------------------------
        # Risk Factors
        # -------------------------------------------------

        risk_factors = shap_df[
            shap_df["SHAP Value"] > 0
        ]


        # -------------------------------------------------
        # Protective Factors
        # -------------------------------------------------

        protective_factors = shap_df[
            shap_df["SHAP Value"] < 0
        ]


        col1, col2 = st.columns(2)


        # -------------------------------------------------
        # Risk Factors Display
        # -------------------------------------------------

        with col1:

            st.markdown(
                "### 🔴 Risk Factors"
            )


            if len(risk_factors) > 0:

                for _, row in risk_factors.iterrows():

                    st.write(
                        f"• **{row['Feature']}** "
                        f"({row['SHAP Value']:.4f})"
                    )

            else:

                st.write(
                    "No major risk factors."
                )


        # -------------------------------------------------
        # Protective Factors Display
        # -------------------------------------------------

        with col2:

            st.markdown(
                "### 🟢 Protective Factors"
            )


            if len(protective_factors) > 0:

                for _, row in protective_factors.iterrows():

                    st.write(
                        f"• **{row['Feature']}** "
                        f"({row['SHAP Value']:.4f})"
                    )

            else:

                st.write(
                    "No protective factors identified."
                )