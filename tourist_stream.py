import streamlit as st
import joblib
import pandas as pd

# Load the saved models
regressor = joblib.load("regressor_model.pkl")
classifier = joblib.load("classifier_model.pkl")

# Streamlit app title
st.title("Tourism Experience Prediction")

# Input section
st.header("Input Parameters")

# User inputs
visit_year = st.number_input("Visit Year", min_value=2000, max_value=2030, step=1, value=2022)
visit_month = st.number_input("Visit Month", min_value=1, max_value=12, step=1, value=1)
attraction_id = st.number_input("Attraction ID", min_value=1, step=1, value=1)

# Prediction button
if st.button("Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        "VisitYear": [visit_year],
        "VisitMonth": [visit_month],
        "AttractionId": [attraction_id]
    })
    
    # Predict using the regression model
    rating_prediction = regressor.predict(input_data)[0]
    # Predict using the classification model
    visit_mode_prediction = classifier.predict(input_data)[0]

    # Display predictions
    st.subheader("Prediction Results")
    st.write(f"Predicted Rating: {rating_prediction:.2f}")
    st.write(f"Predicted Visit Mode: {visit_mode_prediction}")

# Footer
st.text("Model trained using historical tourism data.")
