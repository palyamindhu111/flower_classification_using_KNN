import base64
from pathlib import Path

import joblib
import streamlit as st

# Page settings
st.set_page_config(page_title="Flower Classification", page_icon="🌸")

st.title("🌸 Flower Classification")
st.write("Enter the flower measurements to predict its species.")

MODEL_PATH = Path(__file__).with_name("iris_model.joblib")
BACKGROUND_PATH = Path(__file__).with_name("flower-background.svg")

background_image = base64.b64encode(BACKGROUND_PATH.read_bytes()).decode("ascii")
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/svg+xml;base64,{background_image}") center / cover fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.35);
    }}
    [data-testid="stAppViewContainer"] {{
        background: transparent;
    }}
    .block-container {{
        max-width: 760px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        padding: 2.5rem 3rem;
        border: 1px solid rgba(255, 255, 255, 0.75);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 18px 55px rgba(90, 57, 68, 0.16);
        backdrop-filter: blur(12px);
        color: #24202a;
    }}
    .stApp, .stApp p, .stApp label, .stApp li,
    [data-testid="stMarkdownContainer"] {{
        color: #24202a;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
    }}
    h1, h2, h3 {{
        color: #54243b !important;
        font-family: "Segoe UI", Arial, sans-serif !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em;
    }}
    [data-testid="stWidgetLabel"] p {{
        color: #342733 !important;
        font-size: 1rem !important;
        font-weight: 650 !important;
    }}
    [data-testid="stNumberInput"] input {{
        color: #201820 !important;
        background-color: #fff !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: 1px solid #a68b99 !important;
    }}
    [data-testid="stNumberInput"] button {{
        color: #54243b !important;
    }}
    [data-testid="stAlert"] p {{
        color: #24202a !important;
        font-size: 1rem !important;
    }}
    .stButton > button {{
        border: 0;
        border-radius: 12px;
        background: linear-gradient(90deg, #a83e68, #873254);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        transition: transform 150ms ease, box-shadow 150ms ease;
    }}
    .stButton > button:hover {{
        color: white;
        border: 0;
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(135, 50, 84, 0.32);
    }}
    [data-testid="stCaptionContainer"] {{
        color: #49404a !important;
        font-size: 0.9rem !important;
    }}
    @media (max-width: 640px) {{
        .block-container {{
            margin: 1rem;
            padding: 1.5rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error("Trained model not found. Run the `ml code` script first.")
    st.stop()

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
target_names = model_data["target_names"]

st.subheader("Flower Measurements (cm)")

sepal_length = st.number_input("Sepal length", min_value=0.1, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Flower"):
    values = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(values)
    flower_name = target_names[prediction[0]]

    st.success(f"Predicted flower: **{flower_name.title()}**")
    st.info("This prediction is made by a K-Nearest Neighbors (KNN) model trained on the Iris dataset.")

st.caption("Algorithm: K-Nearest Neighbors (KNN) | Type: Supervised Machine Learning")
