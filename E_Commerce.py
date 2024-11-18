import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Streamlit app title
st.title("E-Commerce Shipment Delivery Prediction")

# Load the model and scaler
try:
    model = joblib.load('shipment_delivery_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'shipment_delivery_model.pkl' and 'scaler.pkl' are in the working directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input fields for user in the left column
with col1:
    id_input = st.number_input('ID (Unique Identifier for the customer)', min_value=0, step=1, format="%d")
    customer_care_calls = st.number_input('Customer Care Calls (Number)', min_value=0, step=1)
    customer_rating = st.number_input('Customer Rating (1 to 5)', min_value=1, max_value=5, step=1)
    cost_of_product = st.number_input('Cost of the Product ($)', min_value=0.0, format="%.2f")

# Input fields for user in the right column
with col2:
    prior_purchases = st.number_input('Prior Purchases (Number)', min_value=0, step=1)
    product_importance = st.selectbox('Product Importance (0=High, 1=Low, 2=Medium)', options=[0, 1, 2])
    discount_offered = st.number_input('Discount Offered ($)', min_value=0.0, format="%.2f")
    weight_in_gms = st.number_input('Weight in grams', min_value=0.0, format="%.2f")

# Dynamic calculations
cost_per_gram = cost_of_product / weight_in_gms if weight_in_gms > 0 else 0
effective_cost = cost_of_product - discount_offered

# Create input DataFrame, excluding 'Gender' and 'Mode_of_Shipment'
input_data = pd.DataFrame({
    'ID': [id_input],  # Include the ID as input
    'Customer_care_calls': [customer_care_calls],
    'Customer_rating': [customer_rating],
    'Cost_of_the_Product': [cost_of_product],
    'Prior_purchases': [prior_purchases],
    'Product_importance': [product_importance],
    'Discount_offered': [discount_offered],
    'Weight_in_gms': [weight_in_gms],
    'Cost_per_gram': [cost_per_gram],
    'Effective_cost': [effective_cost]
})

# Load model and make sure columns are in the correct order
model_features = model.feature_names_in_
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Scale columns that require normalization
columns_to_scale = ['Cost_of_the_Product', 'Discount_offered', 'Weight_in_gms', 'Cost_per_gram', 'Effective_cost']
for col in columns_to_scale:
    if col in model_features:
        input_data[col] = scaler.transform(input_data[[col]])

st.write("### Prepared Input Data")
st.write(input_data)  # Display the prepared input data

# Predict button
if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f"## Predicted Delivery Status: {'On Time' if prediction[0] == 0 else 'Not On Time'}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
