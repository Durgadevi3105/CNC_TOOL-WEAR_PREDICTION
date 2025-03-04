import streamlit as st
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model
import pandas as pd

# Set page configuration at the very beginning
st.set_page_config(page_title="Tool Wear Prediction", page_icon="🔧", layout="wide")

# Load trained LSTM model and scaler
MODEL_PATH = "model.h5"
SCALER_PATH = "scaler.pkl"

model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

selected_columns = [
    "material","feedrate","clamp_pressure", "X1_ActualPosition", "Y1_ActualPosition", "Z1_ActualPosition", "X1_CurrentFeedback", "Y1_CurrentFeedback",
    "M1_CURRENT_FEEDRATE", "X1_DCBusVoltage", "X1_OutputPower", "Y1_OutputPower", "S1_OutputPower"
]

sequence_length = 10

# Set up background image
def set_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg_image("cnc.jpg")

# Custom CSS for Sidebar Styling with White, Black, and Blue
sidebar_style = """
     <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0a0f3c, #1e3a8a, #00bfff); /* Dark Blue to Neon Blue */
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .css-1e4r1h6 {
        background-color: transparent;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

menu = ["Home", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.title("🔬 Tool Wear Prediction using LSTM")
    st.markdown(
        """
        🔬 Overview
         Predictive maintenance is revolutionizing manufacturing by minimizing tool failure and optimizing machining processes. This LSTM-based AI model analyzes real-time CNC sensor data to predict tool wear, ensuring efficiency and precision in machining operations.

        ⚙️ How It Works?
        1️⃣ Real-Time Data Collection 📊
            Captures sensor readings (spindle speed, vibration, temperature, force).
        2️⃣ Data Preprocessing & Feature Engineering 🧩
            Normalizes values, removes noise, and extracts critical features.
        3️⃣ LSTM Model for Pattern Recognition 🧠
            Uses sequential data to detect wear progression trends.
        4️⃣ Prediction & Decision Support ✅
            Provides insights into tool condition and remaining lifespan.
        5️⃣ Integration with Smart Manufacturing Systems 🌍
            Sends alerts for tool replacement, reducing unplanned downtimes.
        📈 Key Benefits
           🔹 Prevents Costly Failures – Reduces sudden tool breakage and machine damage.
           🔹 Optimizes Tool Usage – Ensures tools are replaced at the right time.
           🔹 Enhances Product Quality – Minimizes defects caused by worn-out tools.
           🔹 Boosts Production Efficiency – Reduces machining interruptions and improves workflow.
           """
    )

elif choice == "Prediction":
    st.title("🎨 Tool Wear Prediction")
    st.markdown("Fill in the details below and predict tool wear conditions.")
    
    st.sidebar.header("📥 Enter Feature Values")
    user_input = {feature: st.sidebar.number_input(f"{feature}", value=0.0) for feature in selected_columns}

    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    input_sequence = np.array([input_scaled] * sequence_length)
    input_sequence = input_sequence.reshape(1, sequence_length, len(selected_columns))

    if st.sidebar.button("Predict"):
        pred_tool_condition, pred_machining_finalized, pred_visual_inspection = model.predict(input_sequence)

        tool_condition_labels = {0: "Good", 1: "Worn", 2: "Damaged"}
        pred_tool_condition_label = tool_condition_labels[np.argmax(pred_tool_condition[0])]
        machining_finalized_status = "Yes" if pred_machining_finalized[0][0] > 0.5 else "No"
        visual_inspection_status = "Passed" if pred_visual_inspection[0][0] > 0.5 else "Failed"

        st.subheader("🔍 Prediction Results")
        st.write(f"**🛠 Tool Condition:** {pred_tool_condition_label}")
        st.write(f"**🔄 Machining Finalized:** {machining_finalized_status}")
        st.write(f"**👀 Passed Visual Inspection:** {visual_inspection_status}")
        st.success("✅ Prediction Completed!")
