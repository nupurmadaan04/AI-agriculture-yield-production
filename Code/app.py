import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🌾 CropTop – Smart Agriculture Dashboard",
    page_icon="🌱",
    layout="wide"
)

# -------------------- CUSTOM HEADER --------------------

with open("templates/index.html", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)


# -------------------- LOAD DATA & MODEL --------------------

def load_artifacts():
    df = pd.read_csv("../Datasets/Crops_data.csv")
    df.columns = df.columns.str.strip()
    model = load("../Models/rf_model.pkl")      # <-- changed this line
    scaler = load("../Models/scaler.pkl") 
    
    return df, model, scaler

df, model, scaler = load_artifacts()

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("⚙️ Customize View")

crop_sel = st.sidebar.selectbox(
    "Select Crop 🌿",
    sorted([
        "RICE","WHEAT","MAIZE","SORGHUM","PEARL MILLET","BARLEY","CHICKPEA","SESAMUM","GROUNDNUT",
        "PIGEONPEA","RAPSEED & MUSTARD","SUNFLOWER","SAFFLOWER","CASTOR","LINSEED","SOYABEAN",
        "OIL SEEDS","SUGARCANE","COTTON"
    ])
)
crop_year = st.sidebar.selectbox("📅 Year", sorted(df["Year"].unique()))
crop_state = st.sidebar.selectbox("🏞️ State", sorted(df["State Name"].unique()))
metric = st.sidebar.radio("📊 Metric", ["Area", "Yield", "Production"])

# -------------------- FILTER DATA --------------------
df1 = df[(df["Year"] == crop_year) & (df["State Name"] == crop_state)]

area_col = f"{crop_sel} AREA (1000 ha)"
yield_col = f"{crop_sel} YIELD (Kg per ha)"
prod_col = f"{crop_sel} PRODUCTION (1000 tons)"
col_map = {"Area": area_col, "Yield": yield_col, "Production": prod_col}
col = col_map[metric]


# -------------------- SUMMARY METRICS --------------------
total_area = df1[area_col].sum()
avg_yield = df1[yield_col].mean()
total_prod = df1[prod_col].sum()

st.markdown(f"### 🌾 {crop_sel} Insights for {crop_state} ({crop_year})")
col1, col2, col3 = st.columns(3)
col1.metric("Total Area", f"{total_area:,.0f} (1000 ha)")
col2.metric("Avg Yield", f"{avg_yield:,.0f} kg/ha")
col3.metric("Total Production", f"{total_prod:,.0f} tons")

# -------------------- VISUALIZATIONS --------------------
st.markdown("---")

col_a, col_b = st.columns(2)

# District-level chart
with col_a:
    st.subheader(f"{metric} by District")
    fig = px.bar(
        df1, x="Dist Name", y=col, color="Dist Name",
        title=f"{crop_sel} {metric} Distribution – {crop_state} ({crop_year})",
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(xaxis_title="District", yaxis_title=metric, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Trend over years
with col_b:
    st.subheader(f"Yearly Trend in {crop_state}")
    trend = df[df["State Name"] == crop_state]
    fig2 = px.line(trend, x="Year", y=yield_col, color_discrete_sequence=["#2E8B57"])
    fig2.update_layout(title=f"Yearly Yield Trend for {crop_sel}", xaxis_title="Year", yaxis_title="Yield (Kg/ha)")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------- MODEL PREDICTION SECTION --------------------
st.markdown("---")
st.header("🤖 AI-Powered Rice Crop Yield Prediction")

#'Year', 'RICE AREA (1000 ha)', 'RICE PRODUCTION (1000 tons)', 'State_en']] y=['RICE YIELD (Kg per ha)']

st.markdown("Use the trained ML model to predict **rice crop yield** for custom input values.")

df1 = pd.read_csv("../Datasets/rice_data_outlier_removed.csv")
if model is not None and scaler is not None:
    # User inputs for model prediction
    st.subheader("Enter Crop Parameters")
    col4, col5, col6, col7 = st.columns(4)
    user_state = col4.selectbox("State", sorted(df1["State Name"].unique()), index=0)
    user_year = col5.slider("Year", min_value=int(df1["Year"].min()), max_value=int(df["Year"].max()), value=2020)
    user_area = col6.number_input("Cultivation Area (1000 ha)", min_value=0.0, value=1000.0)
    user_prod = col7.number_input("Production (1000 tons)", min_value=0.0, value=3000.0)

    # Encode state as number
    state_map = {s: i for i, s in enumerate(sorted(df1["State Name"].unique()))}
    state_encoded = state_map[user_state]

    # Prepare input for model
    X = np.array([[user_year, user_area, user_prod, state_encoded]])
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)[0]

    st.success(f"🌿 Predicted Yield for {crop_sel} in {user_state} ({user_year}): **{y_pred:.2f} kg/ha**")

    # Optional visualization
    st.markdown("#### 📈 Comparison with Average Yield")
    avg_state_yield = df[df["State Name"] == user_state][yield_col].mean()

    comp_df = pd.DataFrame({
        "Category": ["Predicted Yield", "Average Yield"],
        "Yield (kg/ha)": [y_pred, avg_state_yield]
    })
    fig3 = px.bar(comp_df, x="Category", y="Yield (kg/ha)", color="Category",
                  color_discrete_sequence=["#00b894", "#0984e3"])
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("⚠️ Model not found. Please place `crop_yield_model.pkl` and `scaler.pkl` inside the `Models/` folder.")