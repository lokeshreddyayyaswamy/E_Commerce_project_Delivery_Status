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
    warehouse_block = st.selectbox('Warehouse Block (0=A, 1=B, 2=C, 3=D, 4=F)', options=[0, 1, 2, 3, 4])
    mode_of_shipment = st.selectbox('Mode of Shipment (0=Flight, 1=Road, 2=Ship)', options=[0, 1, 2])
    customer_care_calls = st.text_input('Customer Care Calls (Number)')
    customer_rating = st.text_input('Customer Rating (1 to 5)')
    cost_of_product = st.text_input('Cost of the Product ($)')

# Input fields for user in the right column
with col2:
    prior_purchases = st.text_input('Prior Purchases (Number)')
    product_importance = st.selectbox('Product Importance (0=High, 1=Low, 2=Medium)', options=[0, 1, 2])
    gender = st.selectbox('Gender (0=Female, 1=Male)', options=[0, 1])
    discount_offered = st.text_input('Discount Offered ($)')
    weight_in_gms = st.text_input('Weight in grams')

# Validate inputs
try:
    customer_care_calls = int(customer_care_calls)
    customer_rating = int(customer_rating)
    cost_of_product = float(cost_of_product)
    prior_purchases = int(prior_purchases)
    discount_offered = float(discount_offered)
    weight_in_gms = float(weight_in_gms)
except ValueError:
    st.error("Please enter valid numeric inputs where required.")
    st.stop()

# Dynamic calculations
cost_per_gram = cost_of_product / weight_in_gms if weight_in_gms > 0 else 0
effective_cost = cost_of_product - discount_offered

# Create input DataFrame
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
    'Weight_in_gms': [weight_in_gms],
    'Cost_per_gram': [cost_per_gram],
    'Effective_cost': [effective_cost]
})

# Ensure all columns match the model's training data
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
