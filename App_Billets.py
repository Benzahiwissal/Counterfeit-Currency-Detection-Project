import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

#  the trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')  # the scaler used during training

# The required features
required_features = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

# Title and description
st.title("Logistic Regression Model Deployment")
st.write("Upload your dataset and get predictions.")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load the uploaded file into a DataFrame
    input_data = pd.read_csv(uploaded_file)

    # Check if all required features are in the uploaded data
    if not all(feature in input_data.columns for feature in required_features):
        st.error("Missing required features!")
    else:
        # Scale the input data (same scaling as during training)
        input_features = input_data[required_features]
        input_features_scaled = scaler.transform(input_features)

        # Make predictions
        predictions = model.predict(input_features_scaled)
        probabilities = model.predict_proba(input_features_scaled)[:, 1]

        # Add predictions and probabilities to the input data
        input_data['Prediction'] = predictions
        input_data['Probability'] = probabilities

        # Display the results
        st.write(input_data[['Id', 'Prediction', 'Probability']])

# Footer
st.write("Model Deployment Complete")
