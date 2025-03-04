import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import base64
def set_image_local(image_path):
    with open(image_path, "rb") as file:
        img = file.read()
    base64_image = base64.b64encode(img).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            #background-position: center;
            #background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_image_local(r"cnc.jpg")

# Load Trained LSTM Model
lstm_model = load_model("D:\streamlit\env\CNC\cnc_model.h5")  

# Title
st.title("CNC Milling Performance Analysis & Fault Detection")

# Task Selection
task = st.selectbox("Select Classification Task", 
                    ["Tool Wear Prediction", "Clamping Detection", "Machining Completion Prediction", "All Predictions"])

# Define feature columns
feature_columns = [
    "feedrate", "clamp_pressure", "material",  
    "M1_CURRENT_FEEDRATE", "X1_ActualPosition", "Y1_ActualPosition",  
    "Z1_ActualPosition", "X1_CurrentFeedback", "Y1_CurrentFeedback",  
    "X1_DCBusVoltage", "X1_OutputPower", "Y1_OutputPower_transformed",  
    "S1_OutputPower"
]

# File uploader
uploaded_file = st.file_uploader("Upload CNC Sensor Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure all required features are present
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing Columns in Uploaded File: {missing_cols}")
    else:
        # Select only necessary columns
        df = df[feature_columns]
        st.write("Uploaded Data Shape:", df.shape)

        # Convert categorical "material" column to numeric if present
        if "material" in df.columns:
            df["material"] = df["material"].astype("category").cat.codes

        # Select only numeric columns before scaling
        numeric_cols = df.select_dtypes(include=["number"]).columns

        # Apply StandardScaler
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[numeric_cols])

        # Reshape for LSTM (samples, timesteps=1, features)
        df_reshaped = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))

        if st.button("Analyze"):
            predictions = lstm_model.predict(df_reshaped)

            # Check output shape of predictions
            st.write("Prediction Output Shape:", predictions.shape)  # Expected: (num_samples, 3)

            if predictions.shape[1] != 3:
                st.error("Error: Model output shape is incorrect! Expected 3 outputs, got:", predictions.shape[1])
            else:
                # Assign predictions to their respective labels
                df["Tool Wear"] = ["Worn" if p[0] > 0.5 else "Unworn" for p in predictions]
                df["Visual inspection"] = ["Properly Clamped" if p[1] > 0.5 else "Not Properly Clamped" for p in predictions]
                df["Machining Completion"] = ["Completed" if p[2] > 0.5 else "Not Completed" for p in predictions]

                st.write(df)

                # Provide download button
                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
