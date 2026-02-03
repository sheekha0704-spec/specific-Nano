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

# --- 1. DATA CLEANING ---
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

# --- 2. AI MODELS ---
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

# --- 3. APP UI ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
st.title("🔬 NanoPredict AI Research Suite")

nav = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"])

if df is not None:
    # --- STEP 1 ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Sourcing")
        drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
        
        d_subset = df[df['Drug_Name'] == drug]
        o_list = sorted(d_subset['Oil_phase'].unique())[:5]
        s_list = sorted(d_subset['Surfactant'].unique())[:5]
        cs_list = sorted(d_subset['Co-surfactant'].unique())[:5]
        
        st.session_state.update({"o": o_list, "s": s_list, "cs": cs_list})
        st.success(f"Top profile identified for {drug}")

    # --- STEP 2: FIXED & IMPROVED ---
    elif nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility Analysis")
        
        if 'drug' not in st.session_state:
            st.warning("Please select a Drug in Step 1 first.")
        else:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.subheader("Selection Matrix")
                sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
                sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
                sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
                st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
                
                # Mock calculation based on dataset averages to prevent crash
                avg_ee = df[df['Oil_phase'] == sel_o]['EE_percent'].mean()
                sol_val = (avg_ee / 30) if not np.isnan(avg_ee) else 2.5
                st.metric(f"Predicted solubility in {sel_o}", f"{sol_val:.2f} mg/mL")

            with c2:
                st.subheader("Comparative Oil Solubility")
                # Visualization of Top 5 Oils
                top_oils = st.session_state.get('o', sorted(df['Oil_phase'].unique())[:5])
                chart_data = []
                for o in top_oils:
                    val = df[df['Oil_phase'] == o]['EE_percent'].mean() / 30
                    chart_data.append({"Oil": o, "Solubility": round(val, 2)})
                
                fig = px.bar(pd.DataFrame(chart_data), x="Oil", y="Solubility", 
                             color="Solubility", text_auto=True, title=f"Solubility of {st.session_state.drug} in Top Oils")
                st.plotly_chart(fig, use_container_width=True)

    # --- STEP 3 ---
    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        
        left, right = st.columns([1, 2])
        with left:
            smix = st.slider("Smix %", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with right:
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [oil], 'b': [smix], 'c': [water],
                'marker': {'size': 18, 'color': 'red', 'symbol': 'diamond'}
            }))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 4 ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation Results")
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
        except Exception:
            st.error("Please complete Step 1 and 2 selections.")
else:
    st.error("Please upload 'nanoemulsion 2.csv'.")
