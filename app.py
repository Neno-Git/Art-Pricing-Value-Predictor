
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained models
reg_pipeline_v2 = joblib.load('catboost_regressor_pipeline_v2.pkl')
clf_pipeline_v2 = joblib.load('high_value_classifier_pipeline_v2.pkl')

# Streamlit App Header
st.title("🎨 Art Pricing & Value Predictor")

# Collecting artwork details
st.subheader('Enter Artwork Details')

# Input fields for artwork data (Updated options)
style = st.selectbox("Style", ['Abstract Expressionism', 'Cubism', 'Impressionism', 'Modern', 'Realism', 'Surrealism'])
medium = st.selectbox("Medium", ['Acrylic', 'Charcoal', 'Mixed Media', 'Oil', 'Watercolour'])
audience = st.selectbox("Target Audience", ['Art Collectors', 'Corporate Clients', 'Families', 'Interior Designers', 'Young Professionals'])

# Input for area in square inches (directly)
area_sq_in = st.number_input('Area (square inches)', min_value=0, value=432)  # Default value

# Automatically assign the cluster based on medium and style
def assign_cluster(medium, style):
    # Define a simple mapping between medium and style to cluster
    cluster_map = {
        # Top 1 - Oil & Acrylic (Cubism/Abstract)
        ('Oil', 'Cubism'): 'Top 1_Oil & Acrylic (Cubism/Abstract)',
        ('Oil', 'Abstract Expressionism'): 'Top 1_Oil & Acrylic (Cubism/Abstract)',
        ('Acrylic', 'Cubism'): 'Top 1_Oil & Acrylic (Cubism/Abstract)',
        ('Acrylic', 'Abstract Expressionism'): 'Top 1_Oil & Acrylic (Cubism/Abstract)',
        
        # Top 2 - Watercolor (Surrealism/Abstract)
        ('Watercolor', 'Surrealism'): 'Top 2_Watercolor (Surrealism/Abstract)',
        ('Watercolor', 'Abstract Expressionism'): 'Top 2_Watercolor (Surrealism/Abstract)',
        
        # Top 5 - Mixed Media (Cubism/Realism)
        ('Mixed Media', 'Cubism'): 'Top 5_Mixed Media (Cubism/Realism)',
        ('Mixed Media', 'Realism'): 'Top 5_Mixed Media (Cubism/Realism)',

        # Default if no match is found
        ('Oil', 'Realism'): 'Other Clusters', 
        ('Acrylic', 'Modern'): 'Other Clusters',
        ('Watercolour', 'Realism'): 'Other Clusters',
        ('Charcoal', 'Surrealism'): 'Other Clusters',
    }
    
    # Return the appropriate cluster based on the combination
    return cluster_map.get((medium, style), 'Other Clusters')

# Automatically assign the cluster based on selected medium and style
assigned_cluster = assign_cluster(medium, style)

# If user clicked the "Predict" button
if st.button('Predict'):
    # Map cluster to a numeric value for the model
    cluster_map = {
        'Top 1_Oil & Acrylic (Cubism/Abstract)': 20,
        'Top 2_Watercolor (Surrealism/Abstract)': 15,
        'Top 3_Oil (Abstract)': 12,
        'Top 4_Charcoal (Surrealism)': 18,
        'Top 5_Mixed Media (Cubism/Realism)': 6,
        'Other Clusters': 0  # Default value for "Other"
    }
    agg_cluster_v2 = cluster_map.get(assigned_cluster, 0)

    sample_data = pd.DataFrame([{
    'style': style,
    'medium': medium,
    'target_audience': audience,
    'agg_cluster': agg_cluster_v2,      # for classification model (V1)
    'agg_cluster_v2': agg_cluster_v2,   # for regression model (V2)
    'area_sq_in': area_sq_in
}])

    # Make the prediction for price (regression)
    predicted_price = reg_pipeline_v2.predict(sample_data)[0]
    
    # Make the classification prediction (high/low value)
    predicted_high_value = clf_pipeline_v2.predict(sample_data)[0]
    value_status = '🏆 High Value Artwork' if predicted_high_value == 1 else '🖼️ Standard Value Artwork'
    
    # Display results with colored background and spacing between the boxes
    st.markdown(f'<div style="background-color: #D4EDDA; padding: 15px; border-radius: 5px; font-size: 16px; color: #155724; margin-bottom: 20px;">💰 Predicted Price: ${predicted_price:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background-color: #B8DAF4; padding: 15px; border-radius: 5px; font-size: 16px; color: #004085;">{value_status}</div>', unsafe_allow_html=True)

