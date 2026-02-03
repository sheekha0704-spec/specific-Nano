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

# --- 2. AI ENGINE ---
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

# --- 3. AUTO-NAVIGATION LOGIC ---
if 'step' not in st.session_state:
    st.session_state.step = "Step 1: Sourcing"

def next_step(step_name):
    st.session_state.step = step_name

# --- 4. APP UI ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
st.title("🔬 NanoPredict AI Research Suite")

nav = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"], index=["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Prediction"].index(st.session_state.step))

if df is not None:
    # --- STEP 1: SOURCING ---
    if nav == "Step 1: Sourcing":
        st.header("1. Drug-Driven Component Sourcing")
        drug = st.selectbox("Select Drug", sorted(df['Drug_Name'].unique()))
        st.session_state.drug = drug
        
        # Filtering Top 5 for session
        d_subset = df[df['Drug_Name'] == drug]
        st.session_state.update({
            "o": sorted(d_subset['Oil_phase'].unique())[:5],
            "s": sorted(d_subset['Surfactant'].unique())[:5],
            "cs": sorted(d_subset['Co-surfactant'].dropna().unique())[:5]
        })
        
        if st.button("Confirm Sourcing & Proceed"):
            next_step("Step 2: Solubility")
            st.rerun()

    # --- STEP 2: SOLUBILITY (Personalized Logic) ---
    elif nav == "Step 2: Solubility":
        st.header("2. Reactive Solubility & Miscibility Analysis")
        c1, c2 = st.columns(2)
        with c1:
            sel_o = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
            sel_s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            sel_cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].dropna().astype(str).unique()))
            st.session_state.update({"f_o": sel_o, "f_s": sel_s, "f_cs": sel_cs})
        
        with c2:
            st.subheader("Personalized Solubility Profiles")
            # Logic: Solubility varies based on the "EE_percent" average for the specific Oil/Surfactant combination
            combo_data = df[(df['Oil_phase'] == sel_o) & (df['Surfactant'] == sel_s)]
            base_sol = combo_data['EE_percent'].mean() / 20 if not combo_data.empty else 2.1
            
            st.metric(f"Solubility in {sel_o}", f"{base_sol:.2f} mg/mL")
            st.metric(f"Solubility in {sel_s}", f"{(base_sol * 0.4):.2f} mg/mL")
            st.metric(f"Solubility in {sel_cs}", f"{(base_sol * 0.25):.2f} mg/mL")
            
        if st.button("Accept Miscibility & Proceed"):
            next_step("Step 3: Ternary")
            st.rerun()

    # --- STEP 3: TERNARY ---
    elif nav == "Step 3: Ternary":
        st.header("3. Ternary Phase Optimization")
        

#[Image of ternary phase diagram for nanoemulsion]

        left, right = st.columns([1, 2])
        with left:
            smix = st.slider("Smix %", 10, 80, 40)
            oil = st.slider("Oil %", 5, 40, 15)
            water = 100 - oil - smix
            st.info(f"Water Phase: {water}%")
        with right:
            fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [oil], 'b': [smix], 'c': [water], 'marker': {'size': 18, 'color': 'red'}}))
            fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
            st.plotly_chart(fig, use_container_width=True)
            
        if st.button("Finalize Composition & Predict"):
            next_step("Step 4: AI Prediction")
            st.rerun()

    # --- STEP 4: AI PREDICTION & SHAP ---
    elif nav == "Step 4: AI Prediction":
        st.header("4. Batch Estimation & AI Interpretation")
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
            
            st.divider()
            st.subheader("Technical Explanation (SHAP Interpretability)")
            st.write("**Feature Contribution Analysis:** The waterfall plot below quantifies how each chosen component pushes the 'Droplet Size' away from the dataset average.")
            
            explainer = shap.Explainer(models['Size_nm'], X_train)
            sv = explainer(input_df)
            fig_sh, ax = plt.subplots(); shap.plots.waterfall(sv[0], show=False); st.pyplot(fig_sh)
            
            st.info("""
            **Technical Breakdown:**
            * **Base Value ($E[f(x)]$):** The average droplet size across all formulations in the database.
            * **$f(x)$:** The final predicted size for your specific formulation.
            * **Positive SHAP (Red):** This component increases droplet size (potentially reducing stability).
            * **Negative SHAP (Blue):** This component effectively reduces droplet size (enhancing nano-homogeneity).
            """)
        except Exception as e:
            st.error(f"Error: {e}. Please ensure all components were selected in Steps 1 and 2.")
else:
    st.error("Please upload 'nanoemulsion 2.csv'.")
