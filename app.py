import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("fraud_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to predict whether the transaction is fraudulent.")

# Transaction type
transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

# Numerical inputs
amount = st.number_input("Amount", min_value=0.0)
old_balance_org = st.number_input("Sender's Old Balance", min_value=0.0)
new_balance_orig = st.number_input("Sender's New Balance", min_value=0.0)
old_balance_dest = st.number_input("Receiver's Old Balance", min_value=0.0)
new_balance_dest = st.number_input("Receiver's New Balance", min_value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "amount": [amount],
        "oldbalanceOrg": [old_balance_org],
        "newbalanceOrig": [new_balance_orig],
        "oldbalanceDest": [old_balance_dest],
        "newbalanceDest": [new_balance_dest],
        "type": [transaction_type]
    })

    # One-hot encode transaction type
    input_data = pd.get_dummies(
        input_data,
        columns=["type"],
        drop_first=True
    )

    # Make sure input has exactly the same columns as training data
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.write(f"Fraud Probability: **{probability:.2%}**")

    if prediction == 1:
        st.error("⚠️ Potential Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")