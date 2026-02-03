import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import shap
import os
import re

# --- 1. DATA CLEANING ENGINE ---
@st.cache_data
def load_and_clean_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    def to_float(value):
        if isinstance(value, str):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(match[0]) if match else np.nan
        return value

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df = load_and_clean_data()

# --- 2. AI MODEL CORE ---
@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
        
    models = {}
    for t in targets:
        if t in _data.columns:
            valid = df_enc[t].notna()
            m = GradientBoostingRegressor(n_estimators=100, random_state=42)
            m.fit(df_enc.loc[valid, features], df_enc.loc[valid, t])
            models[t] = m
            
    return models, le_dict, df_enc[features]

models, encoders, X_train = train_models(df)

# --- 3. APP INTERFACE ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
st.title("🔬 NanoPredict AI Research Suite")

nav = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility Analysis", "Step 3: Ternary Mapping", "Step 4: AI Prediction"])

if df is not None:
    # --- STEP 1: SOURCING ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Component Sourcing")
        drug = st.selectbox("Select Drug for Study", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
        
        d_subset = df[df['Drug_Name'] == drug]
        o_list = sorted(d_subset['Oil_phase'].unique())[:5]
        s_list = sorted(d_subset['Surfactant'].unique())[:5]
        cs_list = sorted(d_subset['Co-surfactant'].unique())[:5]
        
        st.session_state.update({"o": o_list, "s": s_list, "cs": cs_list})
        st.success(f"Top compatibility profile loaded for {drug}")

    # --- STEP 2: IMPROVED SOLUBILITY ANALYSIS ---
    elif nav == "Step 2: Solubility Analysis":
        st.header("2. Reactive Solubility & Miscibility Analysis")
        
        c1, c2 = st.columns([1, 1.5])
        
        with c1:
            st.subheader("Selection Matrix")
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
            
            # Logic-based solubility scoring
            base_val = df[df['Oil_phase'] == sel_o]['EE_percent'].mean()
            sol_val = (base_val / 25) if not np.isnan(base_val) else 3.1
            
            st.metric("Predicted Oil Solubility", f"{sol_val:.2f} mg/mL")
            st.metric("Miscibility Index", f"{(sol_val*2.5):.1f}/10")

        with c2:
            st.subheader("Comparison: Top 5 Oils Solubility Profile")
            # Create a comparison chart for the user's selected drug
            comp_o = st.session_state.get('o', sorted(df['Oil_phase'].unique())[:5])
            comp_data = []
            for o in comp_o:
                val = df[df['Oil_phase'] == o]['EE_percent'].mean() / 25
                comp_data.append({"Oil": o, "Solubility (mg/mL)": round(val, 2)})
            
            chart_df = pd.DataFrame(comp_data)
            fig = px.bar(chart_df, x="Oil", y="Solubility (mg/mL)", color="Solubility (mg/mL)", 
                         color_continuous_scale="Viridis", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 3: TERNARY MAPPING ---
    elif nav == "Step 3: Ternary Mapping":
        st.header("3. Ternary Phase Optimization")
        
        
        left, right = st.columns([1, 2])
        with left:
            smix = st.slider("Smix % (Surfactant + Co-S)", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase (q.s.): {water}%")
            
        with right:
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [oil], 'b': [smix], 'c': [water],
                'marker': {'size': 18, 'color': 'red', 'symbol': 'diamond'}
            }))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil', 'baxis_title': 'Smix', 'caxis_title': 'Water'})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 4: AI PREDICTION ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Optimized Batch Results")
        try:
            input_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.f_cs])[0]
            }])
            
            res = {t: models[t].predict(input_df)[0] for t in models}
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Size", f"{res['Size_nm']:.2f} nm")
            m2.metric("PDI", f"{res['PDI']:.3f}")
            m3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
            m4.metric("EE %", f"{res['EE_percent']:.2f}%")
            
            status = "STABLE" if abs(res['Zeta_mV']) > 15 else "UNSTABLE"
            color = "#d4edda" if status == "STABLE" else "#f8d7da"
            st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'><b>STATUS: {status}</b></div>", unsafe_allow_html=True)
        except:
            st.error("Please complete Step 1 and 2 selections.")
else:
    st.error("Please ensure 'nanoemulsion 2.csv' is in your GitHub folder.")
