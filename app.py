import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load saved objects ────────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# ── App UI ────────────────────────────────────────────
st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to get an estimated selling price.")

brand = st.selectbox("Brand", ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Tata', 'Mahindra'])
year = st.slider("Year", 2000, 2023, 2015)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox("Seller Type", ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
owner = st.selectbox("Owner", ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=18.0)
engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=500.0, value=80.0)
seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

if st.button("Predict Price"):
    # Build input dataframe
    input_df = pd.DataFrame([{
        'year': year,
        'km_driven': np.sqrt(km_driven),
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': np.log1p(max_power),
        'seats': seats,
        'brand': brand
    }])

    # Apply target encoding
    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
    input_encoded = encoder.transform(input_df)

    # Apply scaling
    cols_to_scale = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])

    # Predict
    log_price = model.predict(input_encoded)[0]
    price = np.expm1(log_price)

    st.success(f"Estimated Selling Price: ₹ {price:,.0f}")