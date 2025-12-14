with open('app.py', 'w', encoding='utf-8') as f:
    f.write("""import streamlit as st
import pandas as pd
import numpy as np

st.title("Income Predictor")
st.write("Test app")
""")
print("Файл app.py создан с кодировкой UTF-8")
