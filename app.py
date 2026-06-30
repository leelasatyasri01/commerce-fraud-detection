import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="E-Commerce Fraud Detection",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 E-Commerce Fraud Detection")
st.write("Enter customer and transaction details below.")

customer_age = st.number_input("Customer Age", 18, 100, 30)
customer_tenure = st.number_input("Customer Tenure (Years)", 1, 20, 5)
order_value = st.number_input("Order Value", 100, 50000, 5000)
items = st.number_input("Number of Items", 1, 20, 2)

payment_method = st.selectbox(
    "Payment Method",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "Credit Card",
        1: "Debit Card",
        2: "UPI",
        3: "Cash on Delivery"
    }[x]
)

shipping_distance = st.number_input("Shipping Distance (km)", 1, 3000, 100)

previous_orders = st.number_input("Previous Orders", 0, 100, 10)

failed_payments = st.number_input("Failed Payments", 0, 10, 0)

account_age = st.number_input("Account Age (Months)", 1, 120, 12)

discount_used = st.selectbox("Discount Used", [0, 1])

express_shipping = st.selectbox("Express Shipping", [0, 1])

device_type = st.selectbox(
    "Device Type",
    [0, 1, 2],
    format_func=lambda x: {
        0: "Mobile",
        1: "Desktop",
        2: "Tablet"
    }[x]
)

if st.button("Predict Fraud"):

    data = np.array([[
        customer_age,
        customer_tenure,
        order_value,
        items,
        payment_method,
        shipping_distance,
        previous_orders,
        failed_payments,
        account_age,
        discount_used,
        express_shipping,
        device_type
    ]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")