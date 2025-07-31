import streamlit as st
import pickle
import pandas as pd

with open("models/StandartScaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

with open("models/DecisionTree.pkl", 'rb') as f:
    model = pickle.load(f)

with open("models/LabelEncoder.pkl", 'rb') as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="MycoMind 🍄", page_icon="🍄", layout="centered")
st.title("🍄 MycoMind - Mushroom Classification")
st.write("This app predicts whether a mushroom is **edible** or **poisonous** based on its physical characteristics.")

st.subheader("🔍 Select mushroom features:")

