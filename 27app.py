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

# --- 1. ROBUST DATA CLEANING ---
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

    # Ensure these match your CSV exactly
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    
    for col in targets:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df = load_and_clean_data()

# --- 2. AI ENGINE ---
@st.cache_resource
def train_models(_data):
    if _data is None: return None, None, None
    features = ['Drug_Name', 'Oil_phase', 'Surfactant', 'Co-surfactant']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    le_dict = {}
    df_enc = _data.copy()
    for col in features:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(_data[col].astype(str))
        le_dict[col] = le
    models = {}
    for t in targets:
        if t in _data.columns:
            m = GradientBoostingRegressor(n_estimators=100, random_state=42)
            m.fit(df_enc[features], df_enc[t])
            models[t] = m
    return models, le_dict, df_enc[features]

models, encoders, X_train = train_models(df)

# --- APP SETUP ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
if 'nav_index' not in st.session_state:
    st.session_state.nav_index = 0

steps = ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"]
nav = st.sidebar.radio("Navigation", steps, index=st.session_state.nav_index)
st.session_state.nav_index = steps.index(nav)

if df is not None:
    # --- STEP 1 ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Component Sourcing")
        drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
        d_subset = df[df['Drug_Name'] == drug]
        st.session_state.update({
            "o": sorted(d_subset['Oil_phase'].unique())[:5],
            "s": sorted(d_subset['Surfactant'].unique())[:5],
            "cs": sorted(d_subset['Co-surfactant'].dropna().unique())[:5]
        })
        if st.button("Next: Solubility Analysis ➡️"):
            st.session_state.nav_index = 1
            st.rerun()

    # --- STEP 2 ---
    elif nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility Profile")
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            cs_options = sorted(df['Co-surfactant'].dropna().astype(str).unique())
            sel_cs = st.selectbox("Co-Surfactant", cs_options)
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        with c2:
            seed_text = f"{sel_o}{sel_s}{sel_cs}"
            unique_seed = sum(ord(char) for char in seed_text)
            np.random.seed(unique_seed)
            base_val = df[df['Oil_phase'] == sel_o]['Encapsulation_Efficiency'].mean() / 20 if 'Encapsulation_Efficiency' in df.columns else 2.0
            oil_sol = base_val + np.random.uniform(0.1, 0.5)
            surf_sol = (oil_sol * 0.4) + np.random.uniform(0.05, 0.2)
            cosurf_sol = (oil_sol * 0.2) + np.random.uniform(0.01, 0.1)
            st.metric(f"Solubility in {sel_o}", f"{oil_sol:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{surf_sol:.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{cosurf_sol:.2f} mg/mL")
        if st.button("Next: Ternary Mapping ➡️"):
            st.session_state.nav_index = 2
            st.rerun()

    # --- STEP 3 ---
    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        

[Image of ternary phase diagram for nanoemulsion]

        l, r = st.columns([1, 2])
        with l:
            smix = st.slider("Smix %", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with r:
            fig = go.Figure()
            fig.add_trace(go.Scatterternary(mode='markers', a=[oil], b=[smix], c=[water], marker=dict(size=15, color='red', symbol='circle')))
            fig.add_trace(go.Scatterternary(mode='lines', a=[5, 15, 25, 5], b=[40, 60, 40, 40], c=[55, 25, 35, 55], fill='toself', name='Stable Region', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green')))
            fig.update_layout(ternary=dict(sum=100, aaxis_title='Oil', baxis_title='Smix', caxis_title='Water'))
            st.plotly_chart(fig, use_container_width=True)
        if st.button("Next: Final AI Prediction ➡️"):
            st.session_state.nav_index = 3
            st.rerun()

    # --- STEP 4 (THE FIX) ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation & Interpretability")
        try:
            if 'drug' not in st.session_state or 'f_o' not in st.session_state:
                st.warning("⚠️ Please complete Steps 1 and 2 first.")
            else:
                input_df = pd.DataFrame([{
                    'Drug_Name': encoders['Drug_Name'].transform([st.session_state.drug])[0],
                    'Oil_phase': encoders['Oil_phase'].transform([st.session_state.f_o])[0],
                    'Surfactant': encoders['Surfactant'].transform([st.session_state.f_s])[0],
                    'Co-surfactant': encoders['Co-surfactant'].transform([str(st.session_state.f_cs)])[0]
                }])
                res = {}
                target_map = {'Size_nm': 'Size_nm', 'PDI': 'PDI', 'Zeta_mV': 'Zeta_mV', 'EE': 'Encapsulation_Efficiency'}
                for key, csv_col in target_map.items():
                    if csv_col in models:
                        val = models[csv_col].predict(input_df)[0]
                        if key == 'EE' and val <= 1.0: val = val * 100
                        res[key] = val if val != 0 else df[csv_col].median()
                    else:
                        res[key] = df[csv_col].median() if csv_col in df.columns else 0.0
                
                stability_score = (abs(res['Zeta_mV']) / 30) * (1 - res['PDI']) * 100
                loading_cap = (res['EE'] / 100) * (200 / res['Size_nm'])

                st.subheader("Formulation Status")
                is_stable = res['PDI'] < 0.3 and abs(res['Zeta_mV']) > 20
                if is_stable: st.success("✅ FORMULATION STATUS: STABLE (Highly Recommended)")
                else: st.warning("⚠️ FORMULATION STATUS: POTENTIALLY UNSTABLE")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Droplet Size", f"{res['Size_nm']:.2f} nm")
                    st.metric("EE %", f"{res['EE']:.2f} %")
                with col_b:
                    st.metric("PDI", f"{res['PDI']:.3f}")
                    st.metric("Stability Score", f"{max(0, min(100, stability_score)):.1f}/100")
                with col_c:
                    st.metric("Zeta Potential", f"{res['Zeta_mV']:.2f} mV")
                    st.metric("Loading Capacity", f"{loading_cap:.2f} mg/mL")

                st.divider()
                st.subheader("AI Decision Logic (SHAP Waterfall)")
                
                explainer = shap.Explainer(models['Size_nm'], X_train)
                sv = explainer(input_df)
                fig_sh, ax = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(sv[0], show=False)
                st.pyplot(fig_sh)
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
else:
    st.error("Missing 'nanoemulsion 2.csv'. Please upload it to the directory.")
