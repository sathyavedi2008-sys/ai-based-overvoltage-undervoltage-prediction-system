# ai-based-overvoltage-undervoltage-prediction-system
AI-based system to detect and predict overvoltage and undervoltage using Machine Learning.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Page Configuration
st.set_page_config(
    page_title="AI Voltage Prediction System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ AI-Based Overvoltage and Undervoltage Prediction System")
st.write("Machine Learning Based Voltage Monitoring Dashboard")

# Generate Dataset
@st.cache_resource
def train_model():

    np.random.seed(42)

    voltage = np.random.uniform(150, 280, 5000)
    current = np.random.uniform(1, 20, 5000)
    frequency = np.random.uniform(48, 52, 5000)

    labels = []

    for v in voltage:
        if v < 207:
            labels.append("Undervoltage")
        elif v > 253:
            labels.append("Overvoltage")
        else:
            labels.append("Normal")

    data = pd.DataFrame({
        "Voltage": voltage,
        "Current": current,
        "Frequency": frequency,
        "Status": labels
    })

    X = data[["Voltage", "Current", "Frequency"]]
    y = data["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, data, accuracy, X_test, y_test, predictions


model, data, accuracy, X_test, y_test, predictions = train_model()

# Sidebar
st.sidebar.header("Voltage Input")

voltage = st.sidebar.number_input(
    "Enter Voltage (V)",
    min_value=0.0,
    max_value=500.0,
    value=230.0
)

current = st.sidebar.number_input(
    "Enter Current (A)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

frequency = st.sidebar.number_input(
    "Enter Frequency (Hz)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

# Prediction
st.subheader("Voltage Condition Prediction")

if st.button("Predict Voltage Status"):

    input_data = pd.DataFrame({
        "Voltage": [voltage],
        "Current": [current],
        "Frequency": [frequency]
    })

    result = model.predict(input_data)[0]

    if result == "Normal":
        st.success("NORMAL VOLTAGE")
        st.write("Voltage condition is within the normal range.")

    elif result == "Overvoltage":
        st.error("OVERVOLTAGE DETECTED!")
        st.write("Voltage is above the normal range.")

    else:
        st.warning("UNDERVOLTAGE DETECTED!")
        st.write("Voltage is below the normal range.")

    st.write("Predicted Status:", result)

    probabilities = model.predict_proba(input_data)[0]

    st.write("Prediction Probabilities")

    for name, probability in zip(
        model.classes_, probabilities
    ):
        st.write(f"{name}: {probability * 100:.2f}%")

# Model Accuracy
st.subheader("Machine Learning Model Performance")

st.metric(
    label="Model Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

st.write("Classification Report")

st.text(
    classification_report(y_test, predictions, zero_division=0)
)

# Dataset Display
st.subheader("Voltage Dataset")

st.dataframe(data.head(20))

# Voltage Graph
st.subheader("Voltage Monitoring Graph")

fig, ax = plt.subplots()

ax.plot(data["Voltage"].head(100))
ax.axhline(y=207, linestyle="--", label="Undervoltage Limit")
ax.axhline(y=253, linestyle="--", label="Overvoltage Limit")

ax.set_xlabel("Sample Number")
ax.set_ylabel("Voltage (V)")
ax.set_title("Voltage Monitoring")

ax.legend()

st.pyplot(fig)

# Download Dataset
csv = data.to_csv(index=False)

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name="voltage_dataset.csv",
    mime="text/csv"
)

st.success("AI Voltage Monitoring System is Ready!")
