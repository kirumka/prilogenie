import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Income Predictor')
st.title('Income Prediction App')
st.write('Predict if income > 0K')

try:
    model = joblib.load('model.pkl')
    st.success('Model loaded')

    # 6 features
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider('Age', 17, 90, 35)
        fnlwgt = st.number_input('Final weight', 10000, 1500000, 200000)
        education = st.slider('Education years', 1, 16, 9)

    with col2:
        gain = st.number_input('Capital gain', 0, 100000, 0)
        loss = st.number_input('Capital loss', 0, 5000, 0)
        hours = st.slider('Hours per week', 1, 99, 40)

    if st.button('Predict'):
        data = pd.DataFrame({
            'age': [age],
            'fnlwgt': [fnlwgt],
            'education-num': [education],
            'capital-gain': [gain],
            'capital-loss': [loss],
            'hours-per-week': [hours]
        })

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        if pred == 1:
            st.success('Income > 0K')
        else:
            st.info('Income <= 0K')

        st.write(f'Probability >0K: {prob[1]:.1%}')
        st.write(f'Probability <=0K: {prob[0]:.1%}')

except:
    st.error('Model not found. Run train_model.py first')
