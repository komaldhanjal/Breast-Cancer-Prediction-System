import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== Load =====
model = joblib.load("breast_cancer_model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config("Breast Cancer Prediction", "🩺", layout="centered")
st.title("🩺 Breast Cancer Prediction System")
st.caption("AI-based clinical decision support")

def fix_feature_names(df):
    new_cols = []
    for col in df.columns:
        col = col.lower().strip()

        if "_" in col:
            parts = col.split("_")
            if parts[-1] in ["mean", "se", "worst"]:
                if parts[-1] == "se":
                    parts[-1] = "error"
                col = parts[-1] + " " + " ".join(parts[:-1])
            else:
                col = col.replace("_", " ")

        new_cols.append(col)

    df.columns = new_cols
    return df

uploaded_file = st.file_uploader("📂 Upload Patient CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()], errors="ignore")
    df = fix_feature_names(df)

    if "id" not in df.columns:
        st.error("CSV must contain 'id' column")
        st.stop()

    st.success("File uploaded successfully")

    pid = st.selectbox("Select Patient ID", df["id"].unique())
    patient = df[df["id"] == pid]

    X = patient.drop(columns=["id", "diagnosis"], errors="ignore")

    if st.button("🔍 Predict Cancer"):
        try:
            # ⭐ FINAL SAFETY
            X = X.reindex(columns=scaler.feature_names_in_, fill_value=0)

            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            label = int(np.argmax(pred))

            st.divider()
            st.subheader("Prediction Result")

            if label == 0:
                st.error("❌ Breast Cancer Detected (Malignant)")
            else:
                st.success("✅ No Breast Cancer Detected (Benign)")

        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))
