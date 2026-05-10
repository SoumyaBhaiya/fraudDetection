
# =============================
# 4. STREAMLIT DASHBOARD (FIXED)
# =============================
# Save separately as app.py and RUN LOCALLY

import streamlit as st
import pandas as pd
import numpy as np

import joblib
import time

model = joblib.load("fraud_model.pkl") 

st.title("Real-Time Fraud Detection Dashboard")

columns = [
    "transaction_amount",
    "transaction_time",
    "is_international",
    "device_risk",
    "location_risk",
    "txn_velocity",
    "avg_user_amount"
]

placeholder = st.empty()

while True:
    new_txn = pd.DataFrame([{
        "transaction_amount": np.random.exponential(300),
        "transaction_time": np.random.randint(0,24),
        "is_international": np.random.binomial(1,0.15),
        "device_risk": np.random.uniform(0,1),
        "location_risk": np.random.uniform(0,1),
        "txn_velocity": np.random.randint(1,50),
        "avg_user_amount": np.random.exponential(200)
    }])

    pred = model.predict(new_txn)[0]
    prob = model.predict_proba(new_txn)[0][1]

    with placeholder.container():
        st.subheader("Incoming Transaction")
        st.write(new_txn)

        st.subheader("Prediction")
        st.write("FRAUD" if pred==1 else "LEGIT")
        st.write(f"Fraud Probability: {prob:.2f}")

    time.sleep(5)


# =============================
# 5. HYBRID MODEL IDEA (WRITE IN REPORT)
# =============================
# final_score = RF_probability + IsolationForest_anomaly_score
# if both high → HIGH RISK FRAUD

print("\n✅ PIPELINE READY (RUN STREAMLIT LOCALLY)")
