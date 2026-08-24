
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("xgb_machine_failure_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    air_temperature = float(request.form["air_temperature"])
    process_temperature = float(request.form["process_temperature"])
    rotational_speed = float(request.form["rotational_speed"])
    torque = float(request.form["torque"])
    tool_wear = float(request.form["tool_wear"])

    input_data = pd.DataFrame([[
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear
    ]], columns=[
        "Air temperature K",
        "Process temperature K",
        "Rotational speed rpm",
        "Torque Nm",
        "Tool wear min"
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "Machine Failure Detected"
    else:
        result = "No Machine Failure"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
