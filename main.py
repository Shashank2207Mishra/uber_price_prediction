import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

pipe = pickle.load(open("pipe.pkl", "rb"))
dataset = pickle.load(open("dataset.pkl", "rb"))
st.title("Cab rides price predictor")
distance = st.selectbox("Distance", np.arange(0, 100))
cab_type = st.selectbox('Cab Type', dataset['cab_type'].unique())
Destination = st.selectbox('Destination', dataset["destination"].unique())
source = st.selectbox("Starting_Point", dataset["source"].unique())
surge_multiplier = st.selectbox("Surge_Multiplier", dataset["surge_multiplier"].unique())
Name = st.selectbox("Name_Type", dataset["name"].unique())
if st.button('Predict price'):
    query = np.array([distance, cab_type, Destination, source, surge_multiplier, Name])
    query = query.reshape(1, 6)
    st.subheader("Price in $")
    st.title(pipe.predict(query))