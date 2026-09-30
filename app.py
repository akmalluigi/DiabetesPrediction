import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="",
    layout="wide"
)

st.markdown(f"""
    <style>
    .main{{padding: 0rem 1rem;}}
    .stAlert{{padding: 1rem; border-radius: 0.5rem;}}
    h1{{color: #0072B5; padding-bottom: 1rem;}}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

st.title("Diabetes Prediction")
st.markdown("This application predicts the likelihood of diabetes based on user input features.")

model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Model or scaler files not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are present in the application directory.")
    st.info(""" please run the following command to download the model and scaler files: python diabetes-prediction.py this will train and save the model files,""")
    st.stop()

st.sidebar.title(" Patient Information")
age = st.sidebar.slider("Age", 0, 100, 30)
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 20, 0)

st.sidebar.subheader("Medical Measurements")
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 100)
bp = st.sidebar.slider("Blood Pressure (mmHg)", 0, 130, 70)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level (µU/mL)", 0, 900, 80)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict Diabetes", type="primary", use_container_width=True)

if predict_btn:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    input_std = scaler.transform(input_data)
    prediction = model.predict(input_std)[0]

    try:
        probability = model.predict_proba(input_std)[0]
        prob_negative = probability[0]
        prob_positive = probability[1]
    except:
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive

    st.markdown('---')
    st.header("Prediction Result")

    col1, col2 = st.columns([2,1])

    with col1:
        if prediction == 0:
            if prob_positive < 30:
                st.success(f"The model predicts that the patient is **not likely to have diabetes** with a probability of {prob_negative*100:.2f}%.")
            else:
                st.warning(f"The model predicts that the patient is **not likely to have diabetes** but with a probability of {prob_negative*100:.2f}%. Further testing is recommended.")
        else:
            if prob_positive > 70:
                st.error(f"The model predicts that the patient is **likely to have diabetes** with a probability of {prob_positive*100:.2f}%. Please consult a healthcare professional.")
            else:
                st.warning(f"The model predicts that the patient is **likely to have diabetes** but with a probability of {prob_positive*100:.2f}%. Further testing is recommended.")    

        st.subheader("Probaility breakdown")
        col1, col2 = st.columns(2)
        col1.metric("Probability of No Diabetes", f"{prob_negative*100:.2f}%")
        col2.metric("Probability of Diabetes", f"{prob_positive*100:.2f}%")

    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_positive*100,
            title = {'text': "Diabetes Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if prob_positive > 70 else "orange" if prob_positive > 30 else "green"},
                'steps' : [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold' : {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_positive*100
                }
            }
        ))
        fig.update_layout(height=400, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader("Risk Factors Analysis")

    risk_factors = []
    positive_factors = []

    if glucose > 125:
        risk_factors.append("High Glucose Level (>125 mg/dL)")
    elif glucose < 100:
        positive_factors.append("Low Glucose Level (<100 mg/dL)")

    if bmi > 30:
        risk_factors.append("High BMI (>30)")
    elif bmi < 18.5:
        positive_factors.append("Low BMI (<18.5)")

    if age > 45:
        risk_factors.append("Age over 45")

    if bp > 80:
        risk_factors.append("High Blood Pressure (>80 mmHg)")
    elif 60 <= bp <= 80:
        positive_factors.append("Normal Blood Pressure (60-80 mmHg)")
    else:
        positive_factors.append("Low Blood Pressure (<60 mmHg)")

    if dpf > 0.5:
        risk_factors.append("High Diabetes Pedigree Function (>0.5)")

    if risk_factors:
        st.warning("The following risk factors were identified:")
        for factor in risk_factors:
            st.write(f"- {factor}")

    if positive_factors:
        st.success("The following positive factors were identified:")
        for factor in positive_factors:
            st.write(f"- {factor}")

    st.markdown('---')
    st.subheader("Recommendations")

    if prediction == 1:
        st.error("Based on the prediction, it is recommended to consult a healthcare professional for further evaluation and testing. Lifestyle changes, regular monitoring, and medical advice are crucial.")
    else:
        st.success("Based on the prediction, the patient is not likely to have diabetes. However, maintaining a healthy lifestyle, regular check-ups, and monitoring risk factors are recommended for overall health.")

    st.markdown('---')
    st.warning("Disclaimer: This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
else:
    st.markdown("---")
    st.info("Please input the patient information and click 'Predict Diabetes' to see the prediction results.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Type", "SVM")
    col2.metric("Accuracy", "78%")
    col3.metric("Data Source", "Pima Indians Diabetes Database")
    