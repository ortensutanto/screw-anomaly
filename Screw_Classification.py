import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer

@st.cache_resource
def load_models():
    try:
        with open("Torque_Single_WorkpieceResult.pkl", "rb") as f1:
            workpiece_torque = pickle.load(f1)
        with open("TorqueAngleGradientStep_Multi_WorkpieceResult.pkl", "rb") as f2:
            workpiece_full = pickle.load(f2)
        with open("Torque_Single_ClassValues.pkl", "rb") as f3:
            class_torque = pickle.load(f3)
        with open("TorqueAngleGradientStep_Multi_ClassValues.pkl", "rb") as f4:
            class_full = pickle.load(f4)
            
        return workpiece_torque, workpiece_full, class_torque, class_full
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def pad_values(text):
    tempText = text
    textCount = tempText.count(',')
    diffComma = 83 - int(textCount)

    if(diffComma > 0):
        for i in range(0, diffComma):
            tempText += ',0.0'

    return tempText

def pad_time(text):
    tempText = text 
    inputCount = tempText.count(',')
    splitText = tempText.split(',')
    diff1 = float(splitText[0])
    diff2 = float(splitText[1])
    diff = diff2 - diff1

    diffComma = 83 - inputCount

    lastTime = float(splitText[-1])
    for i in range(0, diffComma):
        lastTime += diff 
        tempText = tempText + "," + str(round(lastTime, 4))

    return tempText

# --- Helper: Convert user input string to pd.Series ---
def parse_input_series(original_time_str, original_value_str):
    try:
        value_str = pad_values(original_value_str)
        time_str = pad_time(original_time_str)
        time = [float(x.strip()) for x in time_str.split(',')]
        values = [float(x.strip()) for x in value_str.split(',')]

        if len(time) != len(values):
            return None, "Number of time points and values must match"

        # EDITED: Return Series with time as index
        return pd.Series(data=values, index=pd.Index(time, name='time')), None
    except ValueError:
        return None, "Invalid number format. Please enter comma-separated numbers"
    except Exception as e:
        return None, f"Error parsing input: {str(e)}"

# --- Helper: Encode categorical variables ---
def encode_categorical(data):
    # Initialize encoders
    location_encoder = LabelBinarizer()
    condition_encoder = LabelBinarizer()

    # Fit and transform location
    location_encoded = location_encoder.fit_transform([[data['workpiece_location']]])[0]
    location_cols = [f'workpiece_location_{c}' for c in ['left', 'middle', 'right']]
    location_dict = dict(zip(location_cols, location_encoded))

    # Fit and transform condition
    condition_encoded = condition_encoder.fit_transform([[data['scenario_condition']]])[0]
    condition_cols = [f'scenario_condition_{c}' for c in ['normal', 'abnormal']]
    condition_dict = dict(zip(condition_cols, condition_encoded))

    # Remove original categorical columns
    del data['workpiece_location']
    del data['scenario_condition']

    # Add encoded columns
    data.update(location_dict)
    data.update(condition_dict)

    return data

# --- Page Config ---
st.set_page_config(page_title="Screw Classification", page_icon="🔩", layout="centered")
st.title("🔩 Screw Classification Inference App")

# Load models at startup
workpiece_torque, workpiece_full = load_models()
if workpiece_torque is None or workpiece_full is None:
    st.error("Failed to load models. Please check if the model files exist and are valid.")
    st.stop()

menu = ["🏠 Home", "🔧 Torque-Only Workpiece Result Classification", "🚰 Custom Feature Workpiece Result Classification", 
                    "🔧 Torque-Only Class Value Classification", "🚰 Custom Feature Class Value Classification",]
choice = st.sidebar.radio("Choose Mode", menu)

# --- Home Page ---
if choice == "🏠 Home":
    st.subheader("Welcome!")
    st.write("This app predicts the screw's workpiece result and class values using either torque alone or multiple sensor features.")
    st.info("Select an option from the sidebar to begin.")

    st.markdown("""
    ### How to Use
    1. Input time series data as comma-separated values
    2. Time values should be in seconds
    3. All features (torque, angle, etc.) should share the same time points
    4. Values should be numeric

    ### Example Input
    Time: 0.0, 0.1, 0.2, 0.3, 0.4
    Torque: 0.1, 0.15, 0.2, 0.25, 0.3
    """)

# --- Torque-Only Workpiece-Result Mode ---
elif choice == "🔧 Torque-Only Workpiece Result Classification":
    st.subheader("🔧 Predict Workpiece Result Using Torque Only")

    st.info("Enter time series data for torque measurements. Values should be comma-separated.")

    time_input = st.text_input("Time Values (comma-separated)", "0.0,0.1,0.2,0.3,0.4")
    torque_input = st.text_input("Torque Values (comma-separated)", "0.1,0.15,0.2,0.25,0.3")

    if st.button("🔍 Predict with Torque"):
        torque_series, error = parse_input_series(time_input, torque_input)

        if error:
            st.error(f"⚠️ {error}")
        else:
            try:
                input_df = pd.DataFrame({"torque": [torque_series]})

                prediction = workpiece_torque.predict(input_df)[0]
                st.success(f"🌟 Predicted Workpiece Result: **{prediction}**")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# --- Workpiece Custom Feature Mode ---
elif choice == "🚰 Custom Feature Workpiece Result Classification":
    st.subheader("🚰 Predict Workpiece Result Using Multiple Features")

    st.markdown("**✅ Select features to use:**")
    use_torque = st.checkbox("Torque", True)
    use_angle = st.checkbox("Angle", True)
    use_gradient = st.checkbox("Gradient", True)
    use_step = st.checkbox("Step", True)
    use_metadata = st.checkbox("Metadata (optional)", True)

    st.markdown("### 📊 Time Series Input")
    st.info("All selected features should share the same time points.")
    time_input = st.text_input("Time Values (shared)", "0.0,0.1,0.2,0.3,0.4")

    features = {}
    has_error = False

    if use_torque:
        torque_input = st.text_input("Torque Values", "0.1,0.15,0.2,0.25,0.3")
        series, error = parse_input_series(time_input, torque_input)
        if error:
            st.error(f"⚠️ Torque input error: {error}")
            has_error = True
        else:
            features["torque"] = series  # EDITED: use pd.Series

    if use_angle:
        angle_input = st.text_input("Angle Values", "2.5,5.0,7.5,10.0,12.5")
        series, error = parse_input_series(time_input, angle_input)
        if error:
            st.error(f"⚠️ Angle input error: {error}")
            has_error = True
        else:
            features["angle"] = series  # EDITED

    if use_gradient:
        gradient_input = st.text_input("Gradient Values", "0.01,0.02,0.03,0.04,0.05")
        series, error = parse_input_series(time_input, gradient_input)
        if error:
            st.error(f"⚠️ Gradient input error: {error}")
            has_error = True
        else:
            features["gradient"] = series  # EDITED

    if use_step:
        step_input = st.text_input("Step Values", "0,0,1,1,1")
        series, error = parse_input_series(time_input, step_input)
        if error:
            st.error(f"⚠️ Step input error: {error}")
            has_error = True
        else:
            features["step"] = series  # EDITED

    if use_metadata:
        st.markdown("### 🧾 Metadata Input")
        features["workpiece_location"] = st.selectbox("Workpiece Location", ["left", "middle", "right"])
        features["workpiece_usage"] = st.selectbox("Workpiece Usage", [0, 1])
        features["workpiece_result"] = st.selectbox("Workpiece Result", ["OK", "NOK"])
        features["scenario_condition"] = st.selectbox("Scenario Condition", ["normal", "abnormal"])
        features["scenario_exception"] = st.selectbox("Scenario Exception", [0, 1])

    if st.button("🔍 Predict with Selected Features"):
        if has_error:
            st.error("⚠️ Please fix the input errors before predicting.")
        else:
            try:
                # Encode categorical metadata
                if use_metadata:
                    features = encode_categorical(features)

                # EDITED: Create input DataFrame with pd.Series for time series columns
                input_df = pd.DataFrame({k: [v] for k, v in features.items()})

                prediction = workpiece_full.predict(input_df)[0]
                st.success(f"🌟 Predicted Workpiece Result: **{prediction}**")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

# --- Torque-Only Class Values Mode ---
elif choice == "🔧 Torque-Only Class Value Classification":
    st.subheader("🔧 Predict Class Values Using Torque Only")

    st.info("Enter time series data for torque measurements. Values should be comma-separated.")

    time_input = st.text_input("Time Values (comma-separated)", "0.0,0.1,0.2,0.3,0.4")
    torque_input = st.text_input("Torque Values (comma-separated)", "0.1,0.15,0.2,0.25,0.3")

    if st.button("🔍 Predict with Torque"):
        torque_series, error = parse_input_series(time_input, torque_input)

        if error:
            st.error(f"⚠️ {error}")
        else:
            try:
                # EDITED: Pass Series directly into DataFrame
                input_df = pd.DataFrame({"torque": [torque_series]})

                prediction = class_torque.predict(input_df)[0]
                st.success(f"🌟 Predicted Workpiece Result: **{prediction}**")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                
# --- Class Custom Feature Mode ---
elif choice == "🚰 Custom Feature Class Value Classification":
st.subheader("🚰 Predict Class Value Using Multiple Features")

st.markdown("**✅ Select features to use:**")
use_torque = st.checkbox("Torque", True)
use_angle = st.checkbox("Angle", True)
use_gradient = st.checkbox("Gradient", True)
use_step = st.checkbox("Step", True)
use_metadata = st.checkbox("Metadata (optional)", True)

st.markdown("### 📊 Time Series Input")
st.info("All selected features should share the same time points.")
time_input = st.text_input("Time Values (shared)", "0.0,0.1,0.2,0.3,0.4")

features = {}
has_error = False

if use_torque:
    torque_input = st.text_input("Torque Values", "0.1,0.15,0.2,0.25,0.3")
    series, error = parse_input_series(time_input, torque_input)
    if error:
        st.error(f"⚠️ Torque input error: {error}")
        has_error = True
    else:
        features["torque"] = series  # EDITED: use pd.Series

if use_angle:
    angle_input = st.text_input("Angle Values", "2.5,5.0,7.5,10.0,12.5")
    series, error = parse_input_series(time_input, angle_input)
    if error:
        st.error(f"⚠️ Angle input error: {error}")
        has_error = True
    else:
        features["angle"] = series  # EDITED

if use_gradient:
    gradient_input = st.text_input("Gradient Values", "0.01,0.02,0.03,0.04,0.05")
    series, error = parse_input_series(time_input, gradient_input)
    if error:
        st.error(f"⚠️ Gradient input error: {error}")
        has_error = True
    else:
        features["gradient"] = series  # EDITED

if use_step:
    step_input = st.text_input("Step Values", "0,0,1,1,1")
    series, error = parse_input_series(time_input, step_input)
    if error:
        st.error(f"⚠️ Step input error: {error}")
        has_error = True
    else:
        features["step"] = series  # EDITED

if use_metadata:
    st.markdown("### 🧾 Metadata Input")
    features["workpiece_location"] = st.selectbox("Workpiece Location", ["left", "middle", "right"])
    features["workpiece_usage"] = st.selectbox("Workpiece Usage", [0, 1])
    features["workpiece_result"] = st.selectbox("Workpiece Result", ["OK", "NOK"])
    features["scenario_condition"] = st.selectbox("Scenario Condition", ["normal", "abnormal"])
    features["scenario_exception"] = st.selectbox("Scenario Exception", [0, 1])

if st.button("🔍 Predict with Selected Features"):
    if has_error:
        st.error("⚠️ Please fix the input errors before predicting.")
    else:
        try:
            # Encode categorical metadata
            if use_metadata:
                features = encode_categorical(features)

            # EDITED: Create input DataFrame with pd.Series for time series columns
            input_df = pd.DataFrame({k: [v] for k, v in features.items()})

            prediction = class_full.predict(input_df)[0]
            st.success(f"🌟 Predicted Workpiece Result: **{prediction}**")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
