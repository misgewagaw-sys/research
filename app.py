#Finall Deployment code for Streamlit app
#python -m streamlit run "C:\Users\hp\Desktop\Crop_Project\app.py"
import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌾",
    layout="wide"
)
st.markdown("""
<style>
.main {
    background-color: #f5fff5;
}

h1 {
    color: green;
    text-align: center;
}

.stButton>button {
    background-color: green;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate conditions to predict the best crop.")

# Load Model & Encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "Random Forest model.pkl"))
le_region = joblib.load(os.path.join(BASE_DIR, "region_encoder.pkl"))
le_crop = joblib.load(os.path.join(BASE_DIR, "crop_encoder.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))  # ✅ already correct

# Inputs
col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("Select Region", le_region.classes_)
    N = st.slider("Nitrogen (N)", 0, 200, 50)
    P = st.slider("Phosphorus (P)", 0, 150, 40)
    K = st.slider("Potassium (K)", 0, 200, 50)

with col2:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    ph = st.slider("pH Level", 0.0, 14.0, 6.5)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 3000.0, 200.0)

# Prediction
if st.button("🌱 Predict Best Crop"):

    try:
        # Encode region
        region_encoded = le_region.transform([region])[0]

        # Prepare input
        input_data = np.array([[N, P, K, temperature, ph, humidity, rainfall, region_encoded]])

        # ✅🔥 APPLY SCALING (THIS IS THE FIX)
        input_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data)

        # Ensure correct format
        prediction = np.array(prediction).flatten()

        # Decode
        crop = le_crop.inverse_transform(prediction.astype(int))
        result = crop[0]

        # Show result
        st.success(f"✅ Recommended Crop: {result}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.write("👨‍💻 Developed with Streamlit | Machine Learning for Smart Agriculture")