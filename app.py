import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ----------------------------
# Load model & preprocessors
# ----------------------------
model = load_model("deepcsat_model.keras")
tfidf = joblib.load("tfidf.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="DeepCSAT Predictor", layout="centered")

st.title("📞 DeepCSAT – Customer Satisfaction Prediction")
st.write("Predict whether a customer is **Satisfied** or **Not Satisfied** based on interaction details.")

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("Customer Interaction Details")

channel_name = st.selectbox(
    "Channel Name",
    ["Inbound", "Outbound", "Outcall"]
)

category = st.selectbox(
    "Issue Category",
    ["Product Queries", "Order Related", "Returns", "Cancellation", "Technical Support"]
)

sub_category = st.selectbox(
    "Sub-Category",
    ["Life Insurance", "Installation/demo", "Reverse Pickup Enquiry",
     "Product Specific Information", "Not Needed"]
)

product_category = st.selectbox(
    "Product Category",
    ["Electronics", "Insurance", "Appliances", "Others"]
)

tenure_bucket = st.selectbox(
    "Agent Tenure",
    ["On Job Training", "<30", "30-60", "60-90", ">90"]
)

agent_shift = st.selectbox(
    "Agent Shift",
    ["Morning", "Evening", "Night"]
)

item_price = st.number_input("Item Price", min_value=0.0)
handling_time = st.number_input("Connected Handling Time (seconds)", min_value=0.0)

remarks = st.text_area("Customer Remarks")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict CSAT"):
    if remarks.strip() == "":
        st.warning("Please enter customer remarks.")
    else:
        # Create DataFrame for structured input
        input_df = pd.DataFrame([{
            "channel_name": channel_name,
            "category": category,
            "Sub-category": sub_category,
            "Product_category": product_category,
            "Tenure Bucket": tenure_bucket,
            "Agent Shift": agent_shift,
            "Item_price": item_price,
            "connected_handling_time": handling_time
        }])

        # Encode categorical columns manually (simple ordinal mapping)
        for col in input_df.columns:
            if input_df[col].dtype == object:
                input_df[col] = input_df[col].astype("category").cat.codes

        # Scale numeric + encoded features
        num_scaled = scaler.transform(input_df)

        # TF-IDF on remarks
        text_vec = tfidf.transform([remarks]).toarray()

        # Combine features
        X_input = np.hstack((num_scaled, text_vec))

        # Prediction
        prediction = model.predict(X_input)[0][0]

        if prediction > 0.5:
            st.error("❌ Customer is NOT Satisfied")
        else:
            st.success("✅ Customer is Satisfied")

        st.write(f"**Confidence Score:** {prediction:.2f}")
