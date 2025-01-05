import streamlit as st
import joblib
import pandas as pd

# the trained model
model = joblib.load('best_model.pkl')

# the required features
required_features = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

# Title and description
st.title("Logistic Regression Model Deployment")
st.write("Upload your dataset and get predictions.")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # the uploaded file into a DataFrame
    input_data = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write("Uploaded Data:")
    st.dataframe(input_data)

    try:
        # Select only the required features
        input_features = input_data[required_features]

        # if the input features are numerical
        for column in input_features.columns:
            if input_features[column].dtype == 'object':
                st.write(f"Encoding column: {column}")
                input_features[column] = pd.factorize(input_features[column])[0]

        # Make predictions
        predictions = model.predict(input_features)
        probabilities = model.predict_proba(input_features)[:, 1]

        # Display results
        st.write("Predictions:")
        input_data['Prediction'] = predictions
        input_data['Probability'] = probabilities
        st.dataframe(input_data[['Id', 'Prediction', 'Probability']])

    except KeyError as e:
        st.error(f"Missing required feature(s): {e}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Footer
st.write("Model Deployment Complete")
