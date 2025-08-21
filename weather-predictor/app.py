import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("weather_model.pkl")
print("Model expects features:", model.feature_names_in_)

# Streamlit app UI
st.title("🌤️ Weather Forecast Predictor")

humidity = st.number_input("Humidity (%)", value=60.0)
pressure = st.number_input("Pressure (hPa)", value=1013.0)
wind_speed = st.number_input("Wind Speed (m/s)", value=5.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Humidity (%)": humidity,
        "Pressure (hPa)": pressure,
        "Wind Speed (m/s)": wind_speed
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"🌡️ Predicted Temperature: {prediction:.2f} °C")
