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
gill_attachment = st.selectbox("Gill attachment", ["attached", "descending", 'free', 'notched'])
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
habitat = st.selectbox("habitat", ["grasses", 'leaves', 'meadows', 'paths', 'urban','waste','woods'])

cap_shape_dict = {"bell":"b", 'conical':'c', "convex":'x', 'flat': 'f', "knobbed": 'k', 'sunken':'s'}
cap_surface_dict = {"fibrous": "f", 'grooves':'g', 'scaly':'y', 'smooth': 's'}
cap_color_dict = {"brown": 'n', 'buff':"b",'cinnamon':'c', 'gray':'g', 'green': 'r', 'pink':'p', 'purple':'u', 'red': 'e','white':'w', 'yellow':'y'}
bruises_dict = {'bruises':'t', 'no':'f'}
odor_dict = {"almond":'a', 'anise':'l', 'creosote': 'c', 'fishy':'y', 'foul':'f', 'musty':'m', 'none': 'n', 'pungent': 'p','spicy':'s'}
gill_attachment_dict = {'attached': 'a', 'descending': 'd', 'free':'f', 'notched':'n'}
gill_spacing_dict = {'close': 'c', 'crowded': 'w', 'distant': 'd'}
gill_size_dict = {'broad':'b', 'narrow':'n'}
gill_color_dict = {"black": 'k', 'brown':'n','buff':'b', 'chocolate':'h', 'gray':'g', 'green':'r', 'orange':'o', 'pink':'p', 'purple':'u', 'red':'e', 'white':'w', 'yellow':'y'}
stalk_shape_dict = {"enlarging":'e', 'tapering':'t'}
stalk_root_dict = {"bulbous":'b', 'club':'c', 'cup':'u', 'equal':'e', 'rhizomorphs':'z', 'rooted':'r'}
stalk_surface_above_ring_dict = {"fibrous":'f', 'scaly':'y', 'silky':'k', 'smooth':'s'}
stalk_surface_below_ring_dict = {"fibrous":'f', 'scaly':'y', 'silky':'k', 'smooth':'s'}
stalk_color_above_ring_dict = {"brown":'n', 'buff':'b', 'cinnamon':'c', 'gray':'g', 'orange':'o', 'pink':'p', 'red':'e', 'white':'w', 'yellow':'y'}
stalk_color_below_ring_dict = {"brown":'n', 'buff':'b', 'cinnamon':'c', 'gray':'g', 'orange':'o', 'pink':'p', 'red':'e', 'white':'w', 'yellow':'y'}
veil_type_dict = {'partial':'p', 'universal':'u'}
veil_color_dict = {'brown':'n', 'orange':'o', 'white':'w', 'yellow':'y'}
ring_number_dict = {"none":'n', 'one':'o', 'two':'t'}
ring_type_dict = {'cobwebby':'c', 'evanescent':'e', 'flaring':'f', 'large':'l', 'none':'n', 'pendant':'p', 'sheathing':'s', 'zone':'z'}
spore_print_color_dict = {'black':'k', 'brown':'n', 'buff':'b', 'chocolate':'h', 'green':'r', 'orange':'o', 'purple':'u', 'white':'w', 'yellow':'y'}
population_dict = {'abundant':'a', 'clustered':'c', 'numerous':'n', 'scattered':'s', 'several':'v', 'solitary':'y'}
habitat_dict = {'grasses':'g', 'leaves':'l', 'meadows':'m', 'paths':'p', 'urban':'u', 'waste':'w', 'woods':'d'}

data = {'cap-shape':cap_shape_dict[cap_shape],
        'cap-surface':cap_surface_dict[cap_surface], 
        'cap-color':cap_color_dict[cap_color],
        'bruises':bruises_dict[bruises], 
        'odor':odor_dict[odor], 
        'gill-attachment':gill_attachment_dict[gill_attachment],
        'gill-spacing':gill_spacing_dict[gill_spacing], 
        'gill-size':gill_size_dict[gill_size], 
        'gill-color':gill_color_dict[gill_color], 
        'stalk-shape':stalk_shape_dict[stalk_shape], 
        'stalk-root':stalk_root_dict[stalk_root], 
        'stalk-surface-above-ring':stalk_surface_above_ring_dict[stalk_surface_above_ring],
        'stalk-surface-below-ring':stalk_surface_below_ring_dict[stalk_surface_below_ring], 
        'stalk-color-above-ring':stalk_color_above_ring_dict[stalk_color_above_ring], 
        'stalk-color-below-ring':stalk_color_below_ring_dict[stalk_color_below_ring], 
        'veil-type':veil_type_dict[veil_type],
        'veil-color':veil_color_dict[veil_color], 
        'ring-number':ring_number_dict[ring_number], 
        'ring-type':ring_type_dict[ring_type], 
        'spore-print-color':spore_print_color_dict[spore_print_color], 
        'population':population_dict[population], 
        'habitat':habitat_dict[habitat]}

if st.button("Predict"):
    for col, le in label_encoders.items():
        if col == 'class':
            continue
        data[col] = le.transform([data[col]])[0]

    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)[0]

    if prediction == 0:
        st.success("✅ The mushroom is **edible**.")
    else:
        st.error("❌ Warning: The mushroom is **poisonous**.")