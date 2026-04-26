import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_model():
    with open("kmeans_model.pkl", "rb") as file:
        return pickle.load(file)


FEATURES = [
    "annual_income",
    "purchase_amount",
    "purchase_frequency",
    "loyalty_score",
]

st.title("Customer Segmentation with K-Means")
st.write(
    "Upload the customer dataset to fit the scaler, then enter a single customer to predict a segment."
)

model = load_model()

st.subheader("1) Upload dataset for scaling")
file = st.file_uploader("Upload Customer Purchasing Behaviors CSV", type=["csv"])
scaler = None

def prepare_scaler(dataframe: pd.DataFrame) -> StandardScaler:
    scaler_local = StandardScaler()
    scaler_local.fit(dataframe[FEATURES])
    return scaler_local

if file is not None:
    df = pd.read_csv(file)
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        df = df.drop_duplicates()
        scaler = prepare_scaler(df)
        st.success("Scaler fitted using uploaded dataset.")
        with st.expander("Preview dataset"):
            st.dataframe(df.head())

st.subheader("2) Enter customer details")
annual_income = st.number_input("Annual income", min_value=0, value=50000, step=1000)
purchase_amount = st.number_input("Purchase amount", min_value=0, value=300, step=10)
purchase_frequency = st.number_input("Purchase frequency", min_value=0, value=15, step=1)
loyalty_score = st.number_input("Loyalty score", min_value=0.0, value=5.0, step=0.1)

input_df = pd.DataFrame(
    {
        "annual_income": [annual_income],
        "purchase_amount": [purchase_amount],
        "purchase_frequency": [purchase_frequency],
        "loyalty_score": [loyalty_score],
    }
)

allow_unscaled = st.checkbox(
    "Allow prediction without scaling (not recommended)", value=False
)

if st.button("Predict segment"):
    if scaler is None and not allow_unscaled:
        st.warning("Upload the dataset first to fit the scaler.")
    else:
        if scaler is None:
            X = input_df.values
        else:
            X = scaler.transform(input_df)
        cluster = int(model.predict(X)[0])
        st.success(f"Predicted segment: Cluster {cluster}")

st.subheader("3) Optional: Segment summary")
if file is not None and scaler is not None:
    df_scaled = scaler.transform(df[FEATURES])
    df["Cluster"] = model.predict(df_scaled)
    st.write(df.groupby("Cluster")[FEATURES].mean())
