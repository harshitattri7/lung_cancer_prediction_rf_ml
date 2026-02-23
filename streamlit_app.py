import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-title {
        color: #FF6B6B;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .positive {
        background-color: #FFE5E5;
        border-left: 4px solid #FF6B6B;
    }
    .negative {
        background-color: #E5F5E5;
        border-left: 4px solid #51CF66;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model files not found! Please ensure models/model.pkl and models/scaler.pkl exist.")
        return None, None

# Title and description
st.markdown('<h1 class="main-title">🫁 Lung Cancer Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered prediction tool using Random Forest classifier</p>', unsafe_allow_html=True)

# Load model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.stop()

# Sidebar for info
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.info("""
    **Model**: Random Forest Classifier
    
    **Accuracy**: 88.71%
    **Recall**: 94.44% (Cancer Detection Rate)
    **AUC-ROC**: 0.9502
    
    *This model was trained on 389 patient records*
    """)
    
    st.markdown("---")
    st.markdown("## 📋 Features Used")
    st.markdown("""
    The model uses 15 clinical and demographic features:
    1. Gender
    2. Age
    3. Smoking Status
    4. Yellow Fingers
    5. Anxiety
    6. Peer Pressure
    7. Chronic Disease
    8. Fatigue
    9. Allergy
    10. Wheezing
    11. Alcohol Consuming
    12. Coughing
    13. Shortness of Breath
    14. Swallowing Difficulty
    15. Chest Pain
    """)

# Create two columns for input
col1, col2 = st.columns(2)

# Define feature names in order (must match training data CSV exactly - 15 features)
feature_names = [
    'GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
    'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ',
    'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
    'SWALLOWING DIFFICULTY', 'CHEST PAIN'
]

# Input form
st.markdown("## 📝 Patient Information")

with col1:
    st.subheader("Demographics & General")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    age = st.slider("Age", 18, 100, 50, key="age")
    
    st.subheader("Symptoms & Risk Factors")
    smoking = st.selectbox("Smoking Status", ["Yes", "No"], key="smoking")
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"], key="yellow_fingers")
    anxiety = st.selectbox("Anxiety", ["Yes", "No"], key="anxiety")
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"], key="peer_pressure")

with col2:
    st.subheader("Medical History")
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"], key="chronic_disease")
    fatigue = st.selectbox("Fatigue", ["Yes", "No"], key="fatigue")
    allergy = st.selectbox("Allergy", ["Yes", "No"], key="allergy")
    
    st.subheader("Current Symptoms")
    wheezing = st.selectbox("Wheezing", ["Yes", "No"], key="wheezing")
    alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"], key="alcohol")
    coughing = st.selectbox("Coughing", ["Yes", "No"], key="coughing")
    shortness = st.selectbox("Shortness of Breath", ["Yes", "No"], key="shortness")
    swallowing = st.selectbox("Swallowing Difficulty", ["Yes", "No"], key="swallowing")
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"], key="chest_pain")

# Prepare data for prediction
yes_no_mapping = {"Yes": 1, "No": 0}
gender_mapping = {"Male": 1, "Female": 0}

input_data = {
    'GENDER': gender_mapping[gender],
    'AGE': age,
    'SMOKING': yes_no_mapping[smoking],
    'YELLOW_FINGERS': yes_no_mapping[yellow_fingers],
    'ANXIETY': yes_no_mapping[anxiety],
    'PEER_PRESSURE': yes_no_mapping[peer_pressure],
    'CHRONIC DISEASE': yes_no_mapping[chronic_disease],
    'FATIGUE ': yes_no_mapping[fatigue],
    'ALLERGY ': yes_no_mapping[allergy],
    'WHEEZING': yes_no_mapping[wheezing],
    'ALCOHOL CONSUMING': yes_no_mapping[alcohol],
    'COUGHING': yes_no_mapping[coughing],
    'SHORTNESS OF BREATH': yes_no_mapping[shortness],
    'SWALLOWING DIFFICULTY': yes_no_mapping[swallowing],
    'CHEST PAIN': yes_no_mapping[chest_pain]
}

# Make prediction
col_btn1, col_btn2 = st.columns([1, 3])

with col_btn1:
    predict_button = st.button("🔍 Analyze Patient", use_container_width=True)

if predict_button:
    # Prepare data in exact order of feature_names
    input_values = [
        input_data['GENDER'],
        input_data['AGE'],
        input_data['SMOKING'],
        input_data['YELLOW_FINGERS'],
        input_data['ANXIETY'],
        input_data['PEER_PRESSURE'],
        input_data['CHRONIC DISEASE'],
        input_data['FATIGUE '],
        input_data['ALLERGY '],
        input_data['WHEEZING'],
        input_data['ALCOHOL CONSUMING'],
        input_data['COUGHING'],
        input_data['SHORTNESS OF BREATH'],
        input_data['SWALLOWING DIFFICULTY'],
        input_data['CHEST PAIN']
    ]
    
    # Create DataFrame with correct column names and order
    df_input = pd.DataFrame([input_values], columns=feature_names)
    X_scaled = scaler.transform(df_input)
    
    # Get predictions
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    # Display result
    st.markdown("---")
    st.markdown("## 🔬 Prediction Results")
    
    if prediction == 1:  # Cancer detected
        st.markdown(f"""
        <div class="prediction-card positive">
            <h3 style="color: #FF6B6B;">⚠️ High Risk - Lung Cancer Suspected</h3>
            <p><strong>Confidence Score:</strong> {probability[1]*100:.2f}%</p>
            <p style="color: #666; font-size: 0.9em;">
            <strong>Recommendation:</strong> This patient shows concerning symptoms. 
            Immediate consultation with a pulmonologist is recommended for further testing and diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:  # No cancer
        st.markdown(f"""
        <div class="prediction-card negative">
            <h3 style="color: #51CF66;">✅ Low Risk - No Cancer Detected</h3>
            <p><strong>Confidence Score:</strong> {probability[0]*100:.2f}%</p>
            <p style="color: #666; font-size: 0.9em;">
            <strong>Note:</strong> This is a screening result. Regular check-ups are still recommended,
            especially for high-risk patients.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show detailed probabilities
    col_prob1, col_prob2 = st.columns(2)
    
    with col_prob1:
        st.metric("No Cancer Risk", f"{probability[0]*100:.2f}%")
    
    with col_prob2:
        st.metric("Cancer Risk", f"{probability[1]*100:.2f}%")
    
    # Show input summary
    with st.expander("📋 Patient Data Summary"):
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': [input_data[feat] for feat in feature_names]
        })
        st.dataframe(summary_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.85em; margin-top: 20px;">
    <p>🏥 <strong>Disclaimer:</strong> This tool is for educational and screening purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Please consult with a healthcare provider for medical guidance.</p>
    
    <p>📚 Model: Random Forest Classifier | Dataset: Survey Lung Cancer | 
    <a href="https://github.com/harshitattri7/lung_cancer_prediction_RF_ML" target="_blank">GitHub Repository</a>
    </p>
</div>
""", unsafe_allow_html=True)
