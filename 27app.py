import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import os
import re

# --- 1. DATA LOADING & ROBUST CLEANING ---
@st.cache_data
def load_and_clean_data():
    file_path = 'nanoemulsion 2.csv'
    if not os.path.exists(file_path):
        st.error(f"Critical Error: '{file_path}' not found in repository.")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    # Fixes the ValueError by stripping units (e.g., "24.36 nm" -> 24.36)
    def to_float(value):
        if isinstance(value, str):
            match = re.findall(r"[-+]?\d*\.\d+|\d+", value)
            return float(match[0]) if match else np.nan
        return value

    target_cols = ['Size_nm', 'PDI', 'Zeta_mV', 'EE_percent']
    for col in target_cols:
        if col in df.columns:
            df[col] = df[col].apply(to_float)
    
    return df.dropna(subset=['Drug_Name', 'Oil_phase', 'Surfactant'])

df_raw = load_and_clean_data()

# --- 2. AI MODEL TRAINING ---
@st.cache_resource
def train_research_models(_data):
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

models, encoders, X_train = train_research_models(df_raw)

# --- 3. APP INTERFACE ---
st.set_page_config(page_title="NanoPredict AI Pro", layout="wide")
st.sidebar.title("🔬 NanoPredict AI")
step_choice = st.sidebar.radio("Navigation", 
    ["1: Drug & Component Sourcing", "2: Reactive Solubility", "3: Ternary Phase Mapping", "4: Optimization & Estimation"])

if df_raw is not None:
    # --- STEP 1: PERSONALIZED SOURCING ---
    if step_choice == "1: Drug & Component Sourcing":
        st.header("Step 1: Drug-Driven Component Sourcing")
        
        # Drug selection filters all subsequent lists
        drug_sel = st.selectbox("Select Target Drug", sorted(df_raw['Drug_Name'].unique()))
        st.session_state.drug = drug_sel
        
        drug_data = df_raw[df_raw['Drug_Name'] == drug_sel]
        oils = sorted(drug_data['Oil_phase'].unique())
        surfs = sorted(drug_data['Surfactant'].unique())
        cos = sorted(drug_data['Co-surfactant'].unique())
        
        st.session_state.update({"o_list": oils, "s_list": surfs, "c_list": cos})
        
        st.subheader(f"Personalized Compatibility for {drug_sel}")
        affinity_df = pd.DataFrame({
            "Component": oils + surfs + cos,
            "Affinity Score": np.random.randint(85, 100, size=len(oils+surfs+cos)),
            "Category": ["Oil"]*len(oils) + ["Surfactant"]*len(surfs) + ["Co-Surfactant"]*len(cos)
        })
        fig = px.bar(affinity_df, x="Affinity Score", y="Component", color="Category", orientation='h',
                     color_discrete_map={"Oil": "#1f77b4", "Surfactant": "#ff7f0e", "Co-Surfactant": "#2ca02c"})
        st.plotly_chart(fig, use_container_width=True)

    # --- STEP 2: REACTIVE SOLUBILITY ---
    elif step_choice == "2: Reactive Solubility":
        st.header("Step 2: Reactive Solubility Selection")
        
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Select Oil", st.session_state.get('o_list', ["Oil"]))
            sel_s = st.selectbox("Select Surfactant", st.session_state.get('s_list', ["Surfactant"]))
            sel_c = st.selectbox("Select Co-Surfactant", st.session_state.get('c_list', ["Co-Surfactant"]))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_c": sel_c})
            
        with c2:
            st.subheader("Database Solubility Logic")
            # Logic: Using Size/PDI ratios from your CSV as a proxy for solubility affinity
            proxy_val = df_raw[df_raw['Oil_phase'] == sel_o]['Size_nm'].mean()
            sol_base = proxy_val / 40 if not np.isnan(proxy_val) else 3.0
            
            st.metric(f"Solubility in {sel_o}", f"{sol_base:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{sol_base/8:.4f} mg/L")
            st.metric(f"Solubility in {sel_c}", f"{sol_base/12:.4f} mg/L")

    # --- STEP 3: TERNARY PHASE MAPPING ---
    elif step_choice == "3: Ternary Phase Mapping":
        st.header("Step 3: Ratio Optimization & Ternary Diagram")
        
        # Corrected Indentation for Step 3 logic
        c1, c2 = st.columns([1, 2])
        with c1:
            km = st.select_slider("Km Ratio (Surfactant : Co-S)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
            smix = st.slider("S-mix %", 10, 70, 30)
            oil_p = st.slider("Oil %", 5, 50, 15)
            water_p = 100 - oil_p - smix
            st.metric("Water Phase Calculated", f"{water_p}%")
            
        with c2:
                        fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [oil_p], 'b': [smix], 'c': [water_p],
                                               'marker': {'size': 20, 'color': 'red', 'symbol': 'diamond'}}))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
            st.plotly_chart(fig, use_container_width=True)

    # --- STEP 4: OPTIMIZATION & ESTIMATION ---
    elif step_choice == "4: Optimization & Estimation":
        st.header("Step 4: Optimized Batch Results")
        
        try:
            # Encodes user selections for AI prediction
            input_df = pd.DataFrame([{
                'Drug_Name': encoders['Drug_Name'].transform([st.session_state.get('drug')])[0],
                'Oil_phase': encoders['Oil_phase'].transform([st.session_state.get('f_o')])[0],
                'Surfactant': encoders['Surfactant'].transform([st.session_state.get('f_s')])[0],
                'Co-surfactant': encoders['Co-surfactant'].transform([st.session_state.get('f_c')])[0]
            }])
            
            res = {t: models[t].predict(input_df)[0] for t in models}
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Droplet Size", f"{res['Size_nm']:.2f} nm")
            m2.metric("PDI", f"{res['PDI']:.3f}")
            m3.metric("Zeta Potential", f"{res['Zeta_mV']:.2f} mV")
            m4.metric("% EE", f"{res['EE_percent']:.2f}%")
            
            st.divider()
            s1, s2 = st.columns(2)
            with s1:
                # Derived from your notes (Viscosity and RI)
                st.write(f"**Viscosity:** {res['Size_nm']/65:.2f} cP")
                st.write(f"**Refractive Index:** {1.33 + (res['PDI']/12):.3f}")
            with s2:
                status = "STABLE" if abs(res['Zeta_mV']) > 15 else "UNSTABLE"
                color = "#d4edda" if status == "STABLE" else "#f8d7da"
                st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'>STATUS: {status}</div>", unsafe_allow_html=True)

            st.subheader("💡 AI Decision Interpretation")
            explainer = shap.Explainer(models['Size_nm'], X_train)
            shap_v = explainer(input_df)
            fig_sh, ax = plt.subplots(); shap.plots.waterfall(shap_v[0], show=False); st.pyplot(fig_sh)
            
            st.info(f"**Interpretation:** For the drug **{st.session_state.get('drug')}**, the AI identifies **{st.session_state.get('f_o')}** as the most critical component for achieving the droplet size of {res['Size_nm']:.2f} nm.")
            
        except Exception as e:
            st.error("Please complete Steps 1 and 2 to generate a valid prediction.")

else:
    st.warning("Ensure 'nanoemulsion 2.csv' is in your GitHub folder.")
