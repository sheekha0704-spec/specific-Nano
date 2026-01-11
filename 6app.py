import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import re
from Bio import Entrez
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp

# --- GLOBAL SETUP ---
st.set_page_config(page_title="NanoPredict AI Lab", layout="wide")
Entrez.email = "your.email@example.com" 

# Initialize Session State to prevent "disappearing" data
if 'drug' not in st.session_state: st.session_state.drug = "Resveratrol"
if 'oil' not in st.session_state: st.session_state.oil = "Oleic Acid"
if 'oil_p' not in st.session_state: st.session_state.oil_p = 10.0
if 'smix_ratio' not in st.session_state: st.session_state.smix_ratio = "2:1"

# --- CORE FUNCTIONS ---
@st.cache_data
def fetch_drug_info(name):
    try:
        compounds = pcp.get_compounds(name, 'name')
        if not compounds:
            cids = pcp.get_cids(name, 'name', 'substance', list_return='flat')
            if cids: compounds = [pcp.Compound.from_cid(cids[0])]
        if compounds:
            c = compounds[0]
            return {"smiles": c.canonical_smiles, "logp": c.xlogp or 3.0, "mw": c.molecular_weight, "cid": c.cid}
    except: return None

# --- PAGE 1: DISCOVERY ---
def page_discovery():
    st.header("🧪 Step 1: Drug Discovery")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.session_state.drug = st.text_input("Target Molecule", st.session_state.drug)
        st.session_state.oil = st.selectbox("Oil Phase", ["Oleic Acid", "Miglyol 812", "Castor Oil"])
    with col2:
        data = fetch_drug_info(st.session_state.drug)
        if data:
            st.image(Draw.MolToImage(Chem.MolFromSmiles(data['smiles']), size=(300, 200)))
            st.success(f"LogP: {data['logp']} | MW: {data['mw']}")
        else:
            st.error("Drug not found. Try a scientific name (e.g., 'Curcumin').")

# --- PAGE 2: DESIGN ---
def page_design():
    st.header("⚖️ Step 2: Smix & Mass Design")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.smix_ratio = st.select_slider("Smix Ratio (S:CoS)", options=["3:1", "2:1", "1:1", "1:2"])
        st.session_state.oil_p = st.slider("Oil %", 5.0, 30.0, st.session_state.oil_p)
    with col2:
        batch = 10.0 # 10g 
        parts = [int(x) for x in st.session_state.smix_ratio.split(':')]
        smix_p = 40.0
        s_g = (parts[0]/sum(parts)) * smix_p * (batch/100)
        cs_g = (parts[1]/sum(parts)) * smix_p * (batch/100)
        
        st.table(pd.DataFrame({
            "Component": ["Drug (1%)", "Oil", "Surfactant", "Cosurfactant", "Water"],
            "Grams (for 10g)": [0.1, batch*(st.session_state.oil_p/100), s_g, cs_g, batch - (0.1 + batch*(st.session_state.oil_p/100) + s_g + cs_g)]
        }))

# --- PAGE 3: STABILITY ---
def page_results():
    st.header("📊 Step 3: Stability Analysis")
    
    # Ternary Diagram Fix: Using Plotly Express Scatter Ternary
    st.subheader("Ternary Phase Diagram")
    oil = st.session_state.oil_p
    smix = 45.0
    water = 100 - (oil + smix)
    
    df_tern = pd.DataFrame({'Oil': [oil], 'Smix': [smix], 'Water': [water]})
    fig_tern = px.scatter_ternary(df_tern, a="Oil", b="Smix", c="Water")
    fig_tern.update_traces(marker=dict(size=20, color="green"))
    st.plotly_chart(fig_tern, use_container_width=True)

    # Sensitivity Graph
    if st.checkbox("Show Sensitivity Analysis"):
        o_range = np.linspace(5, 35, 10)
        size_pred = 100 + (o_range * 1.5)
        st.plotly_chart(px.line(x=o_range, y=size_pred, title="Oil % vs Predicted Size (nm)"))

# --- NAVIGATION ---
pg = st.navigation({
    "Main Lab": [
        st.Page(page_discovery, title="1. Discovery", icon="🧪"),
        st.Page(page_design, title="2. Design", icon="⚖️"),
        st.Page(page_results, title="3. Analysis", icon="📊")
    ]
})
pg.run()
