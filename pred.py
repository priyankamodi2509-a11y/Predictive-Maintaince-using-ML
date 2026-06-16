import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Machine Failure Dashboard",
    page_icon="🏭",
    layout="wide"
)

model = pickle.load(open("rf.pkl", "rb"))
scaler = pickle.load(open("st.pkl", "rb"))
st.markdown("""
<style>
.card {
    border: 2px solid #dcdcdc;
    border-radius: 10px;
    background-color:#f8f9fa;
    text-align:center;
    padding:15px;
    margin:10px;
    justify-content:space-through;
    height:100px;
    color:black;
.card h3{
    margin:0;
}

.card p{
    font-size:16px;
}

    
}
</style>
""", unsafe_allow_html=True)
st.title("🏭 Predictive Maintainance and Machine Failure Prediction System")
st.sidebar.header("⚙️🛠️ Dashboard\n")
st.sidebar.subheader("NAVIGATION\n")
st.sidebar.write("🏠︎ Home")
st.sidebar.write("📊 Dataset Overview")
st.sidebar.write("📉 EDA Visualisations")
st.sidebar.write("💡 Insights and Recommendations")
st.sidebar.write("ⓘ About this project\n")
st.sidebar.text("This project analyzes machine failure prediction using exploratory data analysis, visualization and machine learning. The objective is to identify factors contributing to machine failure and build a predictive model that can support proactive maintenance decisions.")
c1,c2,c3,c4=st.columns(4)
with c1:
    st.markdown("""
    <div class="card" style="background-color:#D5EBF8">
        <p>📚 Total Records</p>
        <h5>136429</h5>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="card" style="background-color:#FFF0F5 ">
        <p>📃 Total Features</p>
        <h5>14</h5>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="card" style="background-color:#FCDADA">
        <p>🚫 Null Values</p>
        <h5>0</h5>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="card" style="background-color:#FFD3AC">
        <p>🎯 Target Variable</p>
        <h5>Machine Failure</h5>
    </div>
    """, unsafe_allow_html=True)
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
st.header("Dataset Overview\n")
co1, co2 = st.columns([3,1])

with co1:
    st.subheader("Dataset Sample")
    st.dataframe(df_train.head())

with co2:
    st.subheader("Missing Values")
    st.dataframe(df_train.isnull().sum().reset_index())
st.header("\n Visualizations\n")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        df_train.corr(numeric_only=True),
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

with col2:
    st.subheader("Pair Plot")

    pair = sns.pairplot(
        df_train.sample(500),
        vars=[
            'Air temperature [K]',
            'Process temperature [K]',
            'Rotational speed [rpm]',
            'Torque [Nm]',
            'Tool wear [min]'
        ],
        hue='Machine failure'
    )

    st.pyplot(pair.fig)
st.header("Machine Failure Prediction")
col1,col2,col3,col4,col5,col6 = st.columns(6)

with col1:
    air_temp = st.number_input("Air temperature [K]")
    process_temp = st.number_input("Process temperature [K]")

with col2:
    rpm = st.number_input("Rotational speed [rpm]")
    torque = st.number_input("Torque [Nm]")

with col3:
    tool_wear = st.number_input("Tool wear [min]")
    machine_type = st.selectbox(
        "Machine Type",
        ["L","M","H"]
    )
with col4:
    twf = st.selectbox("TWF", [0, 1])
    hdf = st.selectbox("HDF", [0, 1])
with col5:
    pwf = st.selectbox("PWF", [0, 1])
    osf = st.selectbox("OSF", [0, 1])
with col6:
    rnf = st.selectbox("RNF", [0, 1])
type_map = {
    "L": 0,
    "M": 1,
    "H": 2
}
type_encoded = type_map[machine_type]
if st.button("Predict"):

    data = np.array([[
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        type_encoded, 
        twf,
        hdf,
        pwf,
        osf,
        rnf
      ]])
    scaled_data=scaler.transform(data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
         st.markdown("""
    <div style="
        background:#ffe5e5;
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:25px;
        color:red;">
        ⚠️ MACHINE FAILURE PREDICTED
    </div>
    """, unsafe_allow_html=True)
    else:
        st.markdown("""
    <div style="
        background:#e6ffe6;
        padding:20px;
        border-radius:10px;
        text-align:center;
        font-size:25px;
        color:green;">
        ✅ NO MACHINE FAILURE
    </div>
    """, unsafe_allow_html=True)
        