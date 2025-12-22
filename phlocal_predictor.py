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
    # Ensure the filename matches your uploaded .pkl file
    data = joblib.load('xgboost_ph_model.pkl')
    return data


try:
    model_data = load_model_data()
    model = model_data['model']
    feature_names = model_data['features']
    imputer = model_data['imputer']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# 2. UI Design (English)
# ==========================================
st.title('🏥 In-hospital Mortality Risk Predictor for Pulmonary Hypertension')
st.markdown("""
This tool utilizes a machine learning model (XGBoost) to estimate the in-hospital mortality risk 
for patients with Pulmonary Hypertension (PH) based on five key clinical parameters.
""")
st.markdown("---")

# Input Section
st.header('Step 1: Patient Clinical Information')
col1, col2, col3 = st.columns(3)

user_inputs = {}
with col1:
    user_inputs['sapsii'] = st.number_input('SAPS II Score', value=30.0, help="Simplified Acute Physiology Score II")
    user_inputs['hr'] = st.number_input('Heart Rate (bpm)', value=80.0)
with col2:
    user_inputs['po2'] = st.number_input('PaO2 (mmHg)', value=100.0, help="Partial Pressure of Oxygen")
    user_inputs['lactate'] = st.number_input('Lactate (mmol/L)', value=1.0)
with col3:
    user_inputs['rdw'] = st.number_input('RDW (%)', value=14.0, help="Red Cell Distribution Width")

# ==========================================
# 3. Prediction & Explanation Logic
# ==========================================
if st.button('Calculate Risk', type='primary'):
    # Create DataFrame and ensure feature order
    input_df = pd.DataFrame([user_inputs])[feature_names]

    # Preprocessing with compatibility check
    try:
        input_df = input_df.astype(float)
        input_filled_values = imputer.transform(input_df)
        input_filled = pd.DataFrame(input_filled_values, columns=feature_names)
    except Exception:
        # Fallback for cloud environment version mismatch
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

    # SHAP Explanation
    st.subheader('Step 3: Clinical Decision Support (SHAP Explanation)')

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_filled)

    # Force Plot Visualization
    fig = plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_filled,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    # Save and display locally to avoid rendering issues on Streamlit Cloud
    plt.savefig("temp_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("temp_force_plot.png")

    st.info("""
    **Interpretation:** - **Red bars (positive SHAP):** Clinical features pushing the prediction toward a HIGHER risk of mortality.
    - **Blue bars (negative SHAP):** Clinical features pushing the prediction toward a LOWER risk (Survival).
    """)

# ==========================================
# 4. Footer & Disclaimer
# ==========================================
st.markdown("""
---
**Disclaimer & Instructions:**
1. This tool is intended for academic and research purposes only and should not be used as the sole basis for clinical diagnosis or treatment.
2. Ensure all input values fall within clinically plausible ranges.
3. Technical Support: Powered by XGBoost and SHAP (SHapley Additive exPlanations).
""")