import streamlit as st
import joblib
import numpy as np

st.set_page_config(layout="wide")

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
mappings = joblib.load("category_mappings.pkl")

def encode(col, value):
    reverse = {v: k for k, v in mappings[col].items()}
    return reverse[value]

st.title("Gaming Sleep Predictor")

age = st.slider("Age", 0, 100, 25, 1)
daily_gaming_hours = st.slider("Daily Gaming Hours", 0.0, 24.0, 2.0, 0.5)

gender = st.radio("Gender", list(mappings["gender"].values()),horizontal=True)

genre = st.selectbox("Game Genre", list(mappings["game_genre"].values()))
game = st.selectbox("Primary Game", list(mappings["primary_game"].values()))
platform = st.selectbox("Platform", list(mappings["gaming_platform"].values()))

if st.button("Predict"):
    input_data = np.array([[
        age,
        encode("gender", gender),
        daily_gaming_hours,
        encode("game_genre", genre),
        encode("primary_game", game),
        encode("gaming_platform", platform)
    ]])

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    if pred[0] == 0:
        st.success("✅ Low Risk: You are not at risk for poor sleep.")
    else:
        st.error("⚠️ High Risk: You may be at risk for poor sleep.")