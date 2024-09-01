pip install joblib
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Streamlit app title
st.title("E-Commerce Shipment Delivery Prediction")

# Load the saved model and scaler
model = joblib.load('shipment_delivery_model.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure this is the correctly fitted scaler file

# Main content layout
st.write("### Enter Input Features for Prediction")

# Create columns for layout
col1, col2 = st.columns(2)

# Input fields
with col1:
    warehouse_block = st.selectbox(
        'Warehouse Block (0=A, 1=B, 2=C, 3=D, 4=F)', 
        options=[0, 1, 2, 3, 4]
    )
    mode_of_shipment = st.selectbox(
        'Mode of Shipment (0=Flight, 1=Road, 2=Ship)', 
        options=[0, 1, 2]
    )
    customer_care_calls = st.number_input(
        'Customer Care Calls', 
        min_value=0, max_value=10, value=4
    )
    customer_rating = st.number_input(
        'Customer Rating (1 to 5)', 
        min_value=1, max_value=5, value=3
    )
    cost_of_product = st.number_input(
        'Cost of the Product ($)', 
        min_value=0, max_value=50000, value=200
    )
    prior_purchases = st.number_input(
        'Prior Purchases', 
        min_value=0, max_value=10, value=3
    )

with col2:
    product_importance = st.selectbox(
        'Product Importance (0=High, 1=Low, 2=Medium)', 
        options=[0, 1, 2]
    )
    gender = st.selectbox(
        'Gender (0=Female, 1=Male)', 
        options=[0, 1]
    )
    discount_offered = st.number_input(
        'Discount Offered (%)', 
        min_value=0, max_value=100, value=20
    )
    weight_in_gms = st.number_input(
        'Weight in grams', 
        min_value=0, max_value=10000, value=2000
    )

# Convert user input to DataFrame
input_data = pd.DataFrame({
    'Warehouse_block': [warehouse_block],
    'Mode_of_Shipment': [mode_of_shipment],
    'Customer_care_calls': [customer_care_calls],
    'Customer_rating': [customer_rating],
    'Cost_of_the_Product': [cost_of_product],
    'Prior_purchases': [prior_purchases],
    'Product_importance': [product_importance],
    'Gender': [gender],
    'Discount_offered': [discount_offered],
    'Weight_in_gms': [weight_in_gms]
})

# Ensure all columns match the model's training data
model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Define columns to scale and initialize the scaler
columns_to_scale = ['Cost_of_the_Product', 'Discount_offered', 'Weight_in_gms']
scaler_features = [col for col in columns_to_scale if col in model_features]

# Initialize scaled data
input_data_scaled = input_data.copy()

# Scale each feature individually
for col in scaler_features:
    try:
        # Reshape column data for the scaler
        input_data_scaled[col] = scaler.transform(input_data[[col]].values.reshape(-1, 1))
    except ValueError as e:
        st.error(f"Error scaling feature '{col}': {e}")
        st.stop()

st.write("### Prepared Input Data")
st.write(input_data_scaled)  # Display the prepared input data

# Predict button
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_data_scaled)
        # Display the prediction result
        st.write(f"## Predicted Delivery Status: {'On Time' if prediction[0] == 0 else 'Not On Time'}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
