import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb  # 新增：用于底层预测补丁

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
    user_inputs['sapsii'] = st.number_input('SAPS II Score', min_value=0.0, max_value=150.0, value=30.0)
    user_inputs['hr'] = st.number_input('Heart Rate (bpm)', min_value=0.0, max_value=250.0, value=80.0)
with col2:
    user_inputs['po2'] = st.number_input('PaO2 (mmHg)', min_value=0.0, max_value=800.0, value=100.0)
    user_inputs['lactate'] = st.number_input('Lactate (mmol/L)', min_value=0.0, max_value=30.0, value=1.0)
with col3:
    user_inputs['rdw'] = st.number_input('RDW (%)', min_value=0.0, max_value=50.0, value=14.0)

# ==========================================
# 3. Prediction & Explanation Logic
# ==========================================
if st.button('Calculate Risk', type='primary'):
    # Create DataFrame and ensure feature order
    input_df = pd.DataFrame([user_inputs])[feature_names]

    # Preprocessing with robust compatibility check
    try:
        input_df = input_df.astype(float)
        if hasattr(imputer, 'transform'):
            input_filled_values = imputer.transform(input_df)
            input_filled = pd.DataFrame(input_filled_values, columns=feature_names)
        else:
            input_filled = input_df
    except Exception:
        input_filled = input_df.fillna(0)

    # Perform Prediction with version compatibility patch
    prediction_prob = 0.0
    try:
        # Standard Sklearn-API call
        prediction_prob = model.predict_proba(input_filled)[0][1]
    except Exception:
        try:
            # Fallback: Directly use Booster for Python 3.12/3.13 compatibility
            dmat = xgb.DMatrix(input_filled)
            # If the booster returns 1D array of probabilities
            raw_pred = model.get_booster().predict(dmat)
            prediction_prob = float(raw_pred[0])
        except Exception as e_final:
            st.error(f"Critical Prediction Error: {e_final}")
            st.stop()

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

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_filled)

        fig = plt.figure(figsize=(12, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0] if isinstance(shap_values, list) else shap_values[0],
            input_filled,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.savefig("temp_force.png", bbox_inches='tight', dpi=300)
        st.image("temp_force.png")

        st.info("**Interpretation:** Red bars increase risk, Blue bars decrease risk.")
    except Exception as e:
        st.warning("Prediction is successful, but SHAP visualizer encountered a version issue.")

# ==========================================
# 4. Footer
# ==========================================
st.markdown("---")
st.caption("For Research Purposes Only.")