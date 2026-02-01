import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from predictor import get_prediction, le_dict  # This imports your saved models

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 15px;}
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 22px; font-weight: 800; color: #1a202c; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; font-weight: 800; font-size: 26px; border: 2px solid;}
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug & Data", "Step 2: Solubility", "Step 3: Component Selection", "Step 4: Ratios", "Step 5: AI Analysis"])

# --- STEP 1: DRUG INPUT ---
if step_choice == "Step 1: Drug & Data":
    st.header("Step 1: Chemical Setup")
    smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
        st.session_state.mw = round(Descriptors.MolWt(mol), 2)
        st.image(Draw.MolToImage(mol, size=(300,250)), caption="Molecular Structure")
        st.write(f"**LogP:** {st.session_state.logp} | **MW:** {st.session_state.mw}")

# --- STEP 3: COMPONENT SELECTION ---
elif step_choice == "Step 3: Component Selection":
    st.header("Step 3: AI Component Selection")
    if le_dict:
        st.session_state.oil_choice = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
        st.session_state.s_final = st.selectbox("Select Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Select Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    else:
        st.error("Model files not found. Please upload .pkl files to GitHub.")

# --- STEP 4: RATIOS ---
elif step_choice == "Step 4: Ratios":
    st.header("Step 4: Composition")
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("Smix %", 10, 60, 30)
    st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    st.info(f"Water Content: {st.session_state.water_p}%")

# --- STEP 5: ANALYSIS (THE ACTUAL PREDICTION) ---
elif step_choice == "Step 5: AI Analysis":
    st.header("Step 5: Final AI Predictions")
    
    if st.button("Run Prediction Engine"):
        # Call the predictor logic we created
        results = get_prediction(
            "Unknown", # Drug Name
            st.session_state.get('oil_choice', 'Unknown'),
            st.session_state.get('s_final', 'Unknown'),
            st.session_state.get('cs_final', 'Unknown'),
            12.0 # Default HLB
        )

        # 1. Stability Display
        status_color = "#d4edda" if results['is_stable'] == 1 else "#f8d7da"
        status_text = "STABLE" if results['is_stable'] == 1 else "UNSTABLE"
        st.markdown(f'<div class="status-box" style="background-color: {status_color};">{status_text}</div>', unsafe_allow_html=True)

        # 2. Metric Cards
        cols = st.columns(4)
        metrics = [('Size (nm)', 'Size_nm'), ('PDI', 'PDI'), ('Zeta (mV)', 'Zeta_mV'), ('EE (%)', 'Encapsulation_Efficiency')]
        
        for i, (label, key) in enumerate(metrics):
            with cols[i]:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='m-label'>{label}</div>
                        <div class='m-value'>{results[key]:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # 3. Ternary Plot
        st.subheader("Formulation Mapping")
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.get('oil_p', 15)],
            'b': [st.session_state.get('smix_p', 30)],
            'c': [st.session_state.get('water_p', 55)],
            'marker': {'size': 20, 'color': 'green'}
        }))
        st.plotly_chart(fig, use_container_width=True)
