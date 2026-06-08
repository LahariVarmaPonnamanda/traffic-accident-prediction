import streamlit as st
import pandas as pd
import pickle
import time

# -----------------------------
# Page Config
st.set_page_config(
    page_title="Traffic Prediction System",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------
# Title
st.title("🚦 Traffic & Accident Prediction System")

st.markdown("""
### 🚀 Smart Traffic Analysis  
Predict **traffic congestion** and **accident risk** using ML models.
""")

# -----------------------------
# Load Models
traffic_model = pickle.load(open('traffic_model.pkl', 'rb'))
accident_model = pickle.load(open('accident_model.pkl', 'rb'))

# -----------------------------
# Sidebar Inputs
st.sidebar.header("🚗 Enter Traffic Details")

# Default values
default_inputs = {
    "Weather": "Clear",
    "Road_Type": "City Road",
    "Time_of_Day": "Morning",
    "Speed_Limit": 0,
    "Number_of_Vehicles": 0,
    "Driver_Alcohol": 0,
    "Accident_Severity": 0,
    "Road_Condition": "Low",
    "Vehicle_Type": "Car",
    "Driver_Age": 16,
    "Driver_Experience": 0,
    "Road_Light_Condition": "Daylight"
}

if "inputs" not in st.session_state:
    st.session_state.inputs = default_inputs.copy()

inputs = st.session_state.inputs

# -----------------------------
# Input Fields
inputs["Weather"] = st.sidebar.selectbox("Weather", ["Clear", "Rainy"])
inputs["Road_Type"] = st.sidebar.selectbox("Road Type", ["City Road", "Rural Road", "Highway"])
inputs["Time_of_Day"] = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

inputs["Speed_Limit"] = st.sidebar.slider("Speed Limit", 0, 200, inputs["Speed_Limit"])
inputs["Number_of_Vehicles"] = st.sidebar.slider("Number of Vehicles", 0, 50, inputs["Number_of_Vehicles"])

inputs["Driver_Alcohol"] = st.sidebar.selectbox("Driver Alcohol", [0,1,2,3,4,5])
inputs["Accident_Severity"] = st.sidebar.selectbox("Accident Severity", [0,1,2,3,4,5])

inputs["Road_Condition"] = st.sidebar.selectbox("Road Condition", ["Low","Moderate","High"])
inputs["Vehicle_Type"] = st.sidebar.selectbox("Vehicle Type", ["Car","Truck","Bus","Other"])

inputs["Driver_Age"] = st.sidebar.slider("Driver Age", 16, 100, inputs["Driver_Age"])
inputs["Driver_Experience"] = st.sidebar.slider("Driver Experience", 0, 50, inputs["Driver_Experience"])

inputs["Road_Light_Condition"] = st.sidebar.selectbox(
    "Road Light Condition", ["Daylight","Artificial Light"]
)

# -----------------------------
# Buttons
col1, col2 = st.columns(2)
predict_clicked = col1.button("🔍 Predict")
reset_clicked = col2.button("🔄 Reset")

# -----------------------------
# Reset FIXED
if reset_clicked:
    st.session_state.inputs = default_inputs.copy()
    st.rerun()   # ✅ FIXED HERE

# -----------------------------
# Prediction
if predict_clicked:

    st.subheader("📋 Input Summary")
    st.write(inputs)

    input_df = pd.DataFrame([inputs])
    input_df = pd.get_dummies(input_df)

    # Align Traffic Model
    traffic_cols = traffic_model.feature_names_in_
    for col in traffic_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df_traffic = input_df[traffic_cols]

    # Align Accident Model
    accident_cols = accident_model.feature_names_in_
    for col in accident_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df_accident = input_df[accident_cols]

    # Loading
    with st.spinner("Analyzing data..."):
        time.sleep(1)

    # Predictions
    traffic_pred = traffic_model.predict(input_df_traffic)[0]
    accident_pred = accident_model.predict(input_df_accident)[0]
    accident_prob = accident_model.predict_proba(input_df_accident)[0][1]

    # Labels
    traffic_labels = {0: "Low", 1: "Medium", 2: "High"}

    # -----------------------------
    # Results
    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Traffic Level", traffic_labels[traffic_pred])

    with col2:
        st.metric("Accident Risk", f"{accident_prob*100:.1f}%")

    # Risk Message
    if accident_prob > 0.7:
        st.error("⚠️ High Risk! Drive carefully.")
    elif accident_prob > 0.4:
        st.warning("⚠️ Moderate Risk")
    else:
        st.success("✅ Safe Conditions")

    # Progress bar
    st.subheader("📈 Accident Probability")
    st.progress(int(accident_prob * 100))

# -----------------------------
# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    background-color: #0e1117;
    color: white;
    font-size: 14px;
}
</style>

<div class="footer">
    👩‍💻 Developed by Lahari Ponnamanda 
</div>
""", unsafe_allow_html=True)