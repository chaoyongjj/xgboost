import streamlit as st
import pandas as pd
import numpy as np
import joblib  # 之前保存用的是joblib，加载也建议用joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置与模型加载
# ==========================================
st.set_page_config(page_title="PH Mortality Predictor", layout="wide")


@st.cache_resource
def load_model_data():
    # 注意：文件名必须与你保存的一致
    data = joblib.load('xgboost_ph_model.pkl')
    return data


model_data = load_model_data()
model = model_data['model']
# 自动从保存的文件中获取训练时的特征顺序（['sapsii', 'hr', 'po2', 'lactate', 'rdw']）
feature_names = model_data['features']
imputer = model_data['imputer']

# ==========================================
# 2. 界面设计
# ==========================================
st.title('🏥 肺动脉高压院内死亡率预测工具')
st.markdown("---")

# 使用列布局美化输入框
st.header('第一步：请输入患者临床信息')
col1, col2, col3 = st.columns(3)

user_inputs = {}
with col1:
    user_inputs['sapsii'] = st.number_input('SAPS II 评分', value=30.0, help="简化急性生理学评分")
    user_inputs['hr'] = st.number_input('心率 (bpm)', value=80.0)
with col2:
    user_inputs['po2'] = st.number_input('氧分压 PO2 (mmHg)', value=100.0)
    user_inputs['lactate'] = st.number_input('乳酸 Lactate (mmol/L)', value=1.0)
with col3:
    user_inputs['rdw'] = st.number_input('红细胞分布宽度 RDW (%)', value=14.0)

# ==========================================
# 3. 预测逻辑
# ==========================================
if st.button('开始计算风险', type='primary'):
    # 1. 转换为DataFrame并确保特征顺序严格一致
    input_df = pd.DataFrame([user_inputs])[feature_names]

    # 2. 缺失值填充（使用保存好的训练集均值）
    input_filled = pd.DataFrame(imputer.transform(input_df), columns=feature_names)

    # 3. 进行预测
    prediction_prob = model.predict_proba(input_filled)[0][1]

    # 4. 显示结果
    st.markdown("---")
    st.header('第二步：预测结果分析')

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="预测院内死亡概率", value=f"{prediction_prob:.2%}")
    with m2:
        if prediction_prob > 0.5:
            st.error("⚠️ 判定：高风险患者")
        else:
            st.success("✅ 判定：低风险患者")

    # 5. SHAP 解释
    st.subheader('第三步：SHAP 临床决策支持（可视化解释）')

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_filled)

    # 绘制SHAP力图 (Force Plot)
    # 注意：matplotlib模式下不需要 initjs
    fig = plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_filled,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    # 自动保存并显示，解决Streamlit直接渲染shap图表的兼容性问题
    plt.savefig("temp_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("temp_force_plot.png")

    st.info("解释：红色代表推高死亡风险的因素，蓝色代表降低风险的因素。")

# ==========================================
# 4. 页脚说明
# ==========================================
st.markdown("""
---
**使用说明与免责声明：**
1. 本工具仅供学术研究参考，不作为最终临床诊断依据。
2. 请确保输入的数据处于合理的临床生理范围内。
3. 技术支持：基于 XGBoost 和 SHAP 解释引擎。
""")