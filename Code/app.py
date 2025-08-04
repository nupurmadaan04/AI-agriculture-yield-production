import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

with open("templates/index.html","r") as f:
    st.markdown(f.read(),unsafe_allow_html=True)


df= pd.read_csv("../Datasets/Crops_data.csv")

st.title(" 🌱 CropTop – Your Farming Dashboard 🌱")
st.write("")
st.write("Which Crop you wanna see ?")
st.text("")
crop_sel= st.selectbox(
    "Select a Crop 🌾",
    ["RICE","WHEAT","MAIZE","SORGHUM","PEARL MILLET","BARLEY","CHICKPEA","SESAMUM","GROUNDNUT","PIGEONPEA","RAPSEED & MUSTARD",
     "SUNFLOWER","SAFFLOWER","CASTOR","LINSEED","SOYABEAN","OIL SEEDS","SUGARCANE","COTTON",]
)

crop_year = st.sidebar.selectbox("Select Year" , sorted(df["Year"].unique()))
crop_state = st.sidebar.selectbox("Select State" , sorted(df["State Name"].unique()))
crop_metrics = st.sidebar.radio("Metric",["Area","Yield","Production"])

df1 = df[(df["Year"]== crop_year) & (df["State Name"] == crop_state)]

crop = crop_sel
area_col = f"{crop} AREA (1000 ha)"
yield_col= f"{crop} YIELD (Kg per ha)"
production_col = f"{crop} PRODUCTION (1000 tons)"

if crop_metrics == "Area":
    col= area_col
elif crop_metrics == "Yield":
    col= yield_col
else:
    col = production_col


st.write("")
st.markdown("Your farming insights are ready! 📊")
st.write("")
fig = px.bar(df1 , x= "Dist Name" , y=col ,title=f"  {crop}  -  {crop_metrics} in  -  {crop_state}  -  ({crop_year})",
             color_discrete_sequence=["#e07a5f"])
st.plotly_chart(fig)

st.write("")
st.write("")
total_area = df1[area_col].sum()
total_yield = df1[yield_col].mean()
total_production= df1[production_col].sum()

col1,col2,col3 =st.columns(3)

col1.metric("Total Area - " ,f"{total_area:,.0f} (1000 ha)")
col2.metric("Total Yield - ", f"{total_yield:,.0f} kg/ha")
col3.metric("Total Production - ",f"{total_production:,.0f} tons")
