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

cap_shape = st.selectbox("Cap shape",["bell", "conical", "convex", "flat", 'knobbed', 'sunken'],)
cap_surface = st.selectbox("Cap Surface", ["fibrous", "grooves", "scaly", "smooth"])
cap_color = st.selectbox("Cap Color", ["brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white", "yellow"])
bruises = st.selectbox("Is there a bruise?", ["bruises", "no"])
odor = st.selectbox("Smell", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
gill_attachment = st.selectbox("Gill attachment", ["attachet", "descinding", 'free', 'notched'])
gill_spacing = st.selectbox("Gill Spacing", ["close", 'crowded', 'distant'])
gill_size = st.selectbox("Gill Size", ["broad", 'narrow'])
gill_color = st.selectbox("Gill Color", ["black", 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
stalk_shape = st.selectbox("Stalk Shape", ["enlarging", "tapering"])
stalk_root = st.selectbox("Stalk Root" ,["bulbous", 'club', 'cup', 'equal', 'rhizomorphs', 'rooted'])
stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", ["fibrous", 'scaly', 'silky', 'smooth'])
stalk_surface_below_ring = st.selectbox("Stalc Surface Below Ring", ["fibrous", 'scaly', 'silky', 'smooth'])
stalk_color_above_ring = st.selectbox("Stalc Color Above Ring", ['brown', 'buff','cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
stalk_color_below_ring = st.selectbox("Stalk Color Below ring", ['brown', 'buff','cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
veil_type = st.selectbox("Veil Type", ["partial", 'universal'])
veil_color = st.selectbox("Veil Color", ["brown", 'orange','white', 'yellow'])
ring_number = st.selectbox("Ring Number", ['none', 'one', 'two'])
ring_type = st.selectbox("Ring Type", ['cobwebby', 'evanescent', 'flaring', 'large', 'none','pendant', 'sheathing', 'zone'])
spore_print_color = st.selectbox("Spore print Color", ["black", 'brown','buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])
population = st.selectbox("Population", ["abundant", 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
habitat = st.selectbox("habitat", ["grasses", 'leaves', 'meadowes', 'paths', 'urban','waste','woods'])
