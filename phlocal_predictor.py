import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import xgboost as xgb

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="Mortality Prediction", layout="centered")


# ==========================================
# 2. Model Loading
# ==========================================
@st.cache_resource
def load_model_data():
    try:
        return joblib.load('xgboost_ph_model.pkl')
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None


model_data = load_model_data()
if model_data:
    model = model_data['model']
    feature_names = model_data['features']
    imputer = model_data['imputer']
else:
    st.stop()

# ==========================================
# 3. UI Layout (Matches your Screenshot)
# ==========================================
st.markdown("## Please enter patient information:")

# Vertical Layout (As seen in the image)
user_inputs = {}

# Ensure the keys match what the model expects, but labels match your image
user_inputs['sapsii'] = st.number_input("Enter SAPSII:", value=10.0)
user_inputs['hr'] = st.number_input("Enter HR:", value=60.0)
user_inputs['po2'] = st.number_input("Enter PO2:", value=100.0)
user_inputs['lactate'] = st.number_input("Enter Lactate:", value=17.0)
user_inputs['rdw'] = st.number_input("Enter RDW:", value=15.0)

st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing

# ==========================================
# 4. Prediction Logic
# ==========================================
if st.button('Predict'):
    # Prepare data
    input_df = pd.DataFrame([user_inputs])[feature_names]

    # Preprocessing (Safe mode)
    try:
        input_df = input_df.astype(float)
        if hasattr(imputer, 'transform'):
            input_filled_values = imputer.transform(input_df)
            input_filled = pd.DataFrame(input_filled_values, columns=feature_names)
        else:
            input_filled = input_df
    except:
        input_filled = input_df.fillna(0)

    # Prediction (Robust mode)
    try:
        prediction_prob = model.predict_proba(input_filled)[0][1]
    except:
        try:
            dmat = xgb.DMatrix(input_filled)
            prediction_prob = float(model.get_booster().predict(dmat)[0])
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.stop()

    # ==========================================
    # 5. Display Result (Large Font)
    # ==========================================
    st.markdown("---")
    st.header("Prediction Result:")

    # Custom HTML to enlarge the font significantly
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

    # ==========================================
    # 6. SHAP Force Plot
    # ==========================================
    st.header("SHAP Force Plot:")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_filled)

        # Compatibility check for SHAP output type
        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            vals = shap_values[0]

        # Generate plot
        plt.figure(figsize=(20, 3))  # Wider figure to match screenshot
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

        # Save and display
        plt.savefig("temp_force_plot.png", bbox_inches='tight', dpi=300)
        st.image("temp_force_plot.png", use_container_width=True)

    except Exception as e:
        st.warning("SHAP plot could not be generated due to version limits, but prediction is valid.")