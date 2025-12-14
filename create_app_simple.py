code = """import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Income Predictor", layout="centered")
st.title("Income Prediction App")
st.write("Predict if income > $50K")

try:
    model = joblib.load("model.pkl")
    st.success("Model loaded!")
    
    age = st.slider("Age", 17, 90, 35)
    hours = st.slider("Hours per week", 1, 99, 40)
    
    if st.button("Predict"):
        sample = pd.DataFrame({
            'age': [age],
            'fnlwgt': [200000],
            'education-num': [10],
            'capital-gain': [0],
            'capital-loss': [0],
            'hours-per-week': [hours]
        })
        
        pred = model.predict(sample)[0]
        if pred == 1:
            st.success("Income > $50K")
        else:
            st.info("Income <= $50K")
            
except:
    st.error("Model not found. Run train_model.py first")
"""

# Сохраняем с правильной кодировкой
with open('app.py', 'wb') as f:
    f.write(code.encode('utf-8'))
    
print("app.py created with UTF-8 encoding")
