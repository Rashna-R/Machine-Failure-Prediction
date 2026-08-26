import streamlit as st
import pandas as pd
import pickle
import shap

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

MODEL_PATH = "xgboost_machine_failure_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("⚙️ Machine Failure Prediction")

st.write(
    "Enter machine operating parameters to predict "
    "whether machine failure may occur."
)

# -------------------------------------------------
# INPUT PARAMETERS
# -------------------------------------------------

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

# -------------------------------------------------
# SHAP TOGGLE
# -------------------------------------------------

show_shap = st.toggle(
    "🔍 Show SHAP Explanation",
    value=False
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.button("Predict"):

    # Input data
    input_data = pd.DataFrame({
        "Air temperature K": [air_temperature],
        "Process temperature K": [process_temperature],
        "Rotational speed rpm": [rotational_speed],
        "Torque Nm": [torque],
        "Tool wear min": [tool_wear]
    })

    # -------------------------------------------------
    # MATCH MODEL FEATURE ORDER
    # -------------------------------------------------

    try:
        model_features = model.get_booster().feature_names

        if model_features:
            input_data = input_data[model_features]

    except Exception:
        pass

    # -------------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------------

    prediction = model.predict(input_data)[0]

    # -------------------------------------------------
    # RESULT
    # -------------------------------------------------

    if prediction == 1:

        st.error("⚠️ Machine Failure Detected")

        st.subheader("🤖 AI Explanation")

        st.write(
            "The machine is predicted to fail based on the given "
            "operating conditions. The current operating parameters "
            "show conditions that may increase the possibility of "
            "machine failure."
        )

        st.subheader("🔧 Recommendation")

        st.write(
            "Inspect the machine and tool condition immediately. "
            "Check for excessive tool wear, abnormal temperature, "
            "torque and rotational speed. Preventive maintenance "
            "is recommended."
        )

    else:

        st.success("✅ No Machine Failure")

        st.subheader("🤖 AI Explanation")

        st.write(
            "The machine is currently predicted to operate without "
            "failure based on the given input conditions."
        )

        st.subheader("🔧 Recommendation")

        st.write(
            "Continue monitoring the machine parameters and perform "
            "regular preventive maintenance."
        )

    # =================================================
    # SHAP EXPLANATION
    # =================================================

    if show_shap:

        st.subheader("📊 SHAP Feature Contribution")

        try:

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(input_data)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]

            # -------------------------------------------------
            # SHAP DATAFRAME
            # -------------------------------------------------

            shap_df = pd.DataFrame({
                "Feature": input_data.columns,
                "SHAP Value": shap_values
            })

            # Absolute value for ranking
            shap_df["Impact"] = shap_df["SHAP Value"].abs()

            # Sort highest impact first
            shap_df = shap_df.sort_values(
                "Impact",
                ascending=False
            ).reset_index(drop=True)

            # -------------------------------------------------
            # MAIN CONTRIBUTING FACTOR
            # -------------------------------------------------

            main_feature = shap_df.iloc[0]["Feature"]
            main_value = shap_df.iloc[0]["SHAP Value"]

            st.info(
                f"🎯 **Main Contributing Factor:** {main_feature}"
            )

            # -------------------------------------------------
            # SHAP TABLE
            # -------------------------------------------------

            display_df = shap_df[
                ["Feature", "SHAP Value"]
            ].copy()

            display_df["Effect"] = display_df["SHAP Value"].apply(
                lambda x:
                "🔴 Increases Failure Risk"
                if x > 0
                else "🟢 Reduces Failure Risk"
                if x < 0
                else "⚪ Very Low Impact"
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # RISK FACTORS
            # -------------------------------------------------

            risk_factors = shap_df[
                shap_df["SHAP Value"] > 0
            ]

            protective_factors = shap_df[
                shap_df["SHAP Value"] < 0
            ]

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🔴 Risk Factors")

                if len(risk_factors) > 0:

                    for _, row in risk_factors.iterrows():

                        st.write(
                            f"• **{row['Feature']}** "
                            f"({row['SHAP Value']:.4f})"
                        )

                else:

                    st.write("No major risk factors.")

            with col2:

                st.markdown("### 🟢 Protective Factors")

                if len(protective_factors) > 0:

                    for _, row in protective_factors.iterrows():

                        st.write(
                            f"• **{row['Feature']}** "
                            f"({row['SHAP Value']:.4f})"
                        )

                else:

                    st.write("No protective factors identified.")

            # -------------------------------------------------
            # SIMPLE SHAP SUMMARY
            # -------------------------------------------------

            st.markdown("### 💡 SHAP Summary")

            if main_value > 0:

                st.write(
                    f"**{main_feature}** has the highest contribution "
                    "towards the current machine failure prediction."
                )

            elif main_value < 0:

                st.write(
                    f"**{main_feature}** has the strongest contribution "
                    "towards reducing the failure prediction."
                )

            else:

                st.write(
                    "The current input parameters have relatively "
                    "low SHAP contribution."
                )

        except Exception as e:

            st.warning(
                "SHAP explanation could not be generated."
            )

            st.code(str(e))
