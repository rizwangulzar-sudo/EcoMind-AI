import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🌱 EcoMind AI: Water Pollution Intelligence")
st.write("Move the sliders to see what the AI predicts.")

# Create sliders on the left side of the screen
st.sidebar.header("Water Test Settings")
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 7.0)
temp = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 30.0)
mud = st.sidebar.slider("Water Muddiness (Turbidity)", 0.0, 100.0, 10.0)
oxy = st.sidebar.slider("Dissolved Oxygen", 0.0, 12.0, 6.0)

# Load the saved AI models
metal_model = joblib.load('../models/metal_model.pkl')
bacteria_model = joblib.load('../models/bacteria_model.pkl')

# Put sliders data into a list for the AI
user_input = np.array([[ph, temp, mud, oxy]])

# Make Predictions
predicted_metal = metal_model.predict(user_input)[0]
predicted_bacteria = bacteria_model.predict(user_input)[0]

# Display results
st.subheader("Results:")
st.write(f"📊 **Predicted Heavy Metal Level:** {predicted_metal:.2f} ppm")

if predicted_bacteria == 1:
    st.error("⚠️ Warning: High Risk of Bacteria Contamination!")
else:
    st.success("✅ Clean: Low Risk of Bacteria Contamination.")
