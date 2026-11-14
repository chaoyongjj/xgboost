from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb


# The threshold was derived from the median predicted risk in the training set
# and fixed before model evaluation in the validation cohorts.
RISK_THRESHOLD = 0.113

INPUT_SPECS = {
    "sapsii": {
        "label": "SAPS II score",
        "unit": "points",
        "min": 11.0,
        "max": 93.0,
        "step": 1.0,
    },
    "hr": {
        "label": "Heart rate",
        "unit": "beats/min",
        "min": 37.0,
        "max": 170.0,
        "step": 1.0,
    },
    "po2": {
        "label": "PaO2",
        "unit": "mmHg",
        "min": 18.0,
        "max": 564.0,
        "step": 1.0,
    },
    "lactate": {
        "label": "Lactate",
        "unit": "mmol/L",
        "min": 0.5,
        "max": 19.8,
        "step": 0.1,
    },
    "rdw": {
        "label": "RDW",
        "unit": "%",
        "min": 11.1,
        "max": 28.4,
        "step": 0.1,
    },
}


st.set_page_config(page_title="PH mortality prediction", layout="centered")


@st.cache_resource
def load_model_data():
    try:
        model_path = Path(__file__).with_name("xgboost_ph_model.pkl")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None


model_data = load_model_data()
if not model_data:
    st.stop()

required_keys = {"model", "features", "metadata"}
if not required_keys.issubset(model_data):
    st.error("The model package is incomplete.")
    st.stop()

model = model_data["model"]
feature_names = model_data["features"]
metadata = model_data.get("metadata", {})
model_version = metadata.get("version", "not specified")
last_updated = metadata.get("last_updated", "not specified")

st.title("Pulmonary Hypertension Mortality Prediction")
st.caption(f"Model package version {model_version} | Last updated: {last_updated}")
st.caption("Research prototype only. This estimate must not replace clinical judgement.")

supported_ranges = "; ".join(
    f"{spec['label']} {spec['min']:g}-{spec['max']:g} {spec['unit']}"
    for spec in INPUT_SPECS.values()
)
st.info(
    "Supported input ranges are based on the observed development-cohort values: "
    f"{supported_ranges}. This calculator does not estimate risk outside these ranges."
)

st.markdown("## Please enter patient information:")
st.caption("All five predictors are mandatory. Missing values are not imputed in clinical use.")

user_inputs = {}
for feature_name, spec in INPUT_SPECS.items():
    user_inputs[feature_name] = st.number_input(
        f"{spec['label']} ({spec['unit']})",
        min_value=float(spec["min"]),
        max_value=float(spec["max"]),
        value=None,
        step=float(spec["step"]),
        help=(
            f"Supported range: {spec['min']:g}-{spec['max']:g} {spec['unit']}. "
            "Values outside the development-cohort range are not accepted."
        ),
        key=f"input_{feature_name}",
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button('Predict'):
    missing_fields = [
        spec["label"]
        for feature_name, spec in INPUT_SPECS.items()
        if user_inputs.get(feature_name) is None
    ]
    if missing_fields:
        st.error(
            "Please enter all five predictors before requesting a prediction: "
            + ", ".join(missing_fields)
            + "."
        )
        st.stop()

    input_filled = pd.DataFrame([user_inputs], columns=feature_names).astype(float)

    try:
        prediction_prob = model.predict_proba(input_filled)[0][1]
    except Exception:
        try:
            dmat = xgb.DMatrix(input_filled)
            prediction_prob = float(model.get_booster().predict(dmat)[0])
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()

    st.markdown("---")
    st.header("Prediction Result:")

    result_text = f"In-hospital mortality probability: {prediction_prob:.2%}"
    st.markdown(f"""
        <div style='
            font-size: 30px;
            font-weight: bold;
            color: #333333;
            background-color: #f0f2f6;
            padding: 20px; 
            border-radius: 10px; 
            margin-bottom: 20px;'>
            {result_text}
        </div>
    """, unsafe_allow_html=True)

    risk_label = "High-risk group" if prediction_prob >= RISK_THRESHOLD else "Lower-risk group"
    st.write(
        f"Risk classification: **{risk_label}** "
        f"(training-derived threshold: {RISK_THRESHOLD:.3f})"
    )

    st.header("SHAP Force Plot:")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_filled)

        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            vals = shap_values[0]

        plt.figure(figsize=(20, 3))
        shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                      (list, np.ndarray)) else explainer.expected_value,
            vals,
            input_filled,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            text_rotation=0
        )
        figure = plt.gcf()
        st.pyplot(figure)
        plt.close(figure)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}. The probability estimate remains valid.")
