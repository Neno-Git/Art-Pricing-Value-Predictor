
import streamlit as st
import pandas as pd
import joblib

# Load both pipelines
reg_pipeline = joblib.load('catboost_regressor_pipeline.pkl')
clf_pipeline = joblib.load('high_value_classifier_pipeline.pkl')

# UI
st.title("🎨 Art Pricing & Value Predictor")

style = st.selectbox("Style", ['Abstract Expressionism', 'Cubism', 'Impressionism', 'Modern', 'Realism', 'Surrealism'])
medium = st.selectbox("Medium", ['Acrylic', 'Charcoal', 'Mixed Media', 'Oil', 'Watercolor'])
audience = st.selectbox("Target Audience", ['Art Collectors', 'Corporate Clients', 'Families', 'Interior Designers', 'Young Professionals'])
size = st.selectbox("Size", ['12\"x16\"', '18\"x24\"', '20\"x30\"', '24\"x36\"', '30\"x4\"'])

if st.button("Predict"):
    try:
        # Convert user input to DataFrame
        user_input = pd.DataFrame([{
            'style': style,
            'medium': medium,
            'target_audience': audience,
            'size': size,
            'agg_cluster': cluster
        }])

        # Predict
        predicted_price = reg_pipeline.predict(user_input)[0]
        predicted_value = clf_pipeline.predict(user_input)[0]

        # Show results
        st.success(f"💰 Predicted Price: ${predicted_price:,.2f}")
        st.info("🏆 High Value Artwork" if predicted_value else "🖼️ Standard Value Artwork")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
