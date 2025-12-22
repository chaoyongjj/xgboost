import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Model Loading
# ==========================================
st.set_page_config(
    page_title="PH Mortality Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model_data():
    # Load the dictionary containing model, features, and imputer
    data = joblib.load('xgboost_ph_model.pkl')
    return data


try:
    model_data = load_model_data()
    model = model_data['model']
    feature_names = model_data['features']
    imputer = model_data['imputer']
except Exception as e:
    st.error(f"Error loading model components: {e}")
    st.stop()

# ==========================================
# 2. UI Design (English)
# ==========================================
st.title('🏥 In-hospital Mortality Risk Predictor for Pulmonary Hypertension')
st.markdown("""
This clinical tool utilizes a machine learning model (XGBoost) to estimate the in-hospital 
mortality risk for patients with Pulmonary Hypertension (PH) based on five key clinical parameters.
""")
st.markdown("---")

# Input Section
st.header('Step 1: Patient Clinical Information')
col1, col2, col3 = st.columns(3)

user_inputs = {}
with col1:
    user_inputs['sapsii'] = st.number_input('SAPS II Score', min_value=0.0, max_value=150.0, value=30.0,
                                            help="Simplified Acute Physiology Score II")
    user_inputs['hr'] = st.number_input('Heart Rate (bpm)', min_value=0.0, max_value=250.0, value=80.0)
with col2:
    user_inputs['po2'] = st.number_input('PaO2 (mmHg)', min_value=0.0, max_value=800.0, value=100.0,
                                         help="Partial Pressure of Oxygen")
    user_inputs['lactate'] = st.number_input('Lactate (mmol/L)', min_value=0.0, max_value=30.0, value=1.0)
with col3:
    user_inputs['rdw'] = st.number_input('RDW (%)', min_value=0.0, max_value=50.0, value=14.0,
                                         help="Red Cell Distribution Width")

# ==========================================
# 3. Prediction & Explanation Logic
# ==========================================
if st.button('Calculate Risk', type='primary'):
    # Create DataFrame and ensure feature order
    input_df = pd.DataFrame([user_inputs])[feature_names]

    # Preprocessing with robust compatibility check for Cloud Environment
    try:
        input_df = input_df.astype(float)
        # Handle cases where imputer might have internal version mismatch
        if hasattr(imputer, 'transform'):
            input_filled_values = imputer.transform(input_df)
            input_filled = pd.DataFrame(input_filled_values, columns=feature_names)
        else:
            input_filled = input_df
    except Exception:
        input_filled = input_df.fillna(0)

    # Perform Prediction
    prediction_prob = model.predict_proba(input_filled)[0][1]

    # Display Results
    st.markdown("---")
    st.header('Step 2: Risk Assessment Analysis')

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Predicted Mortality Probability", value=f"{prediction_prob:.2%}")
    with m2:
        if prediction_prob > 0.5:
            st.error("⚠️ Assessment: High-Risk Patient Group")
        else:
            st.success("✅ Assessment: Low-Risk Patient Group")

    # SHAP Explanation with ValueError handling for XGBoost version differences
    st.subheader('Step 3: Clinical Decision Support (SHAP Explanation)')

    try:
        # Standard initialization
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_filled)

        # Plotting
        fig = plt.figure(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_filled,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig("temp_force.png", bbox_inches='tight', dpi=300)
        st.image("temp_force.png")

        st.info("""
        **Interpretation:** - **Red bars:** Features increasing mortality risk.
        - **Blue bars:** Features decreasing mortality risk (protective).
        """)
    except Exception as e:
        st.warning(
            "SHAP visualizer encountered a version compatibility issue, but the prediction above is still accurate.")
        st.write(f"Technical details: {e}")

# ==========================================
# 4. Footer & Disclaimer
# ==========================================
st.markdown("""
---
**Disclaimer:** This tool is for research purposes only. Decisions should be made by qualified clinicians.
""")