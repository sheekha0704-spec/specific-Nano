import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v18.2", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745;
        text-align: center; margin-bottom: 20px;
    }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Critical Error: {csv_file} not found.")
        st.stop()
    df = pd.read_csv(csv_file)
    
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        
    # Fix for unseen labels: Train only on rows where we have targets, but use ALL categories
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Specified").astype(str)

    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        # We fit on the ENTIRE dataframe to avoid "unseen label" errors
        le.fit(df[col])
        df_train[f'{col}_enc'] = le.transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. STATE MANAGEMENT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
keys = ['drug', 'oil', 'aq', 'oil_p', 'smix_p', 's_final', 'cs_final', 'logp', 'mw', 'water_p']
for k in keys:
    if k not in st.session_state: st.session_state[k] = 0.0 if 'p' in k else None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- NAVIGATION ---
nav = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API & Structural Analysis")
    drug = st.selectbox("Select API", sorted(df['Drug_Name'].unique()))
    oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
    if st.button("Confirm Phases →"):
        st.session_state.drug, st.session_state.oil = drug, oil
        go_to_step("Step 2: Concentrations")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Formulation Ratios")
    st.session_state.oil_p = st.number_input("Oil %", 5.0, 50.0, 15.0)
    st.session_state.smix_p = st.number_input("S-mix %", 5.0, 60.0, 30.0)
    st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    if st.button("Next →"): go_to_step("Step 3: AI Screening")

# --- STEP 3: SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Component Screening")
    st.write(f"Screening surfactants for {st.session_state.oil}...")
    if st.button("Proceed to Final Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION (FIXED TYPEERROR) ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Ingredients")
    # sorted() is safe here because we filled NaNs in load_and_prep
    s_final = st.selectbox("Final Surfactant", sorted(df['Surfactant'].unique()))
    cs_final = st.selectbox("Final Co-Surfactant", sorted(df['Co-surfactant'].unique()))
    if st.button("Execute Final AI Run →"):
        st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
        go_to_step("Step 5: Results")

# --- STEP 5: RESULTS (FIXED SYNTAX & VALUEERROR) ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: AI Suite & Phase Diagram")
    
    # Correctly encode the inputs using le_dict
    idx = pd.DataFrame([[
        le_dict['Drug_Name'].transform([st.session_state.drug])[0],
        le_dict['Oil_phase'].transform([st.session_state.oil])[0],
        le_dict['Surfactant'].transform([st.session_state.s_final])[0],
        le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]
    ]], columns=['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc'])
    
    res = {col: models[col].predict(idx)[0] for col in models}
    
    st.columns(4)[0].metric("Predicted Size", f"{res['Size_nm']:.1f} nm")
    
    t1, t2 = st.tabs(["Ternary Phase Diagram", "Release Profile"])
    
    with t1:
        st.subheader("Formulation Mapping")
        # Ternary Plot Logic
        fig_tern = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.oil_p], 
            'b': [st.session_state.smix_p], 
            'c': [st.session_state.water_p],
            'marker': {'color': '#28a745', 'size': 14, 'line': {'width': 2}}
        }))
        fig_tern.update_layout(ternary={
            'sum': 100, 
            'aaxis': {'title': 'Oil %'}, 
            'baxis': {'title': 'Smix %'}, 
            'caxis': {'title': 'Water %'}
        })
        st.plotly_chart(fig_tern, use_container_width=True)

    if st.button("Reset Formulation"): go_to_step("Step 1: Chemical Setup")
