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

# --- SETUP ---
st.set_page_config(page_title="NanoPredict AI Lab", layout="wide")
Entrez.email = "your.email@example.com"  # NCBI requires an email for API access

# State Management
for key, val in {'drug': 'Paclitaxel', 'oil': 'Castor Oil', 'oil_p': 10.0, 'smix_ratio': '2:1'}.items():
    if key not in st.session_state: st.session_state[key] = val

# --- PUBMED SCRAPER ENGINE ---
def search_pubmed(drug, oil):
    query = f"{drug} AND {oil} AND nanoemulsion AND droplet size"
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        record = Entrez.read(handle)
        ids = record["IdList"]
        if not ids: return []
        
        results = []
        fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="text")
        abstracts = fetch_handle.read().split("\n\n")
        
        for text in abstracts:
            # Look for size patterns like '120 nm' or 'droplet size of 50 nm'
            match = re.search(r"(\d+\.?\d*)\s*(nm|nanometers)", text, re.IGNORECASE)
            if match:
                results.append(f"{match.group(0)} (Found in Literature)")
        return results
    except: return ["No specific size data found in recent abstracts."]

# --- PAGE 1: SEARCH & PUBMED VALIDATION ---
def page_discovery():
    st.title("🧪 Lab 1: Chemical Discovery & Lit-Review")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.session_state.drug = st.text_input("Drug Name", st.session_state.drug)
        st.session_state.oil = st.selectbox("Oil Phase", ["Oleic Acid", "Miglyol 812", "Castor Oil"])
        if st.button("🔍 Cross-Reference PubMed"):
            st.session_state.lit_data = search_pubmed(st.session_state.drug, st.session_state.oil)
    
    with c2:
        # Chemical Visuals
        comp = pcp.get_compounds(st.session_state.drug, 'name')
        if comp:
            st.image(Draw.MolToImage(Chem.MolFromSmiles(comp[0].canonical_smiles), size=(300, 200)))
            st.success(f"LogP: {comp[0].xlogp or 'N/A'}")
        
        if 'lit_data' in st.session_state:
            st.subheader("📚 PubMed Findings")
            for item in st.session_state.lit_data: st.info(item)

# --- PAGE 3: STABILITY & SENSITIVITY ---
def page_results():
    st.title("📊 Lab 3: Stability & Sensitivity Analysis")
    
    # Sensitivity Toggle
    run_sens = st.toggle("Enable Sensitivity Analysis (Oil Variation)")
    if run_sens:
        oil_range = np.linspace(5, 25, 10)
        # Predicted Size: Base 110nm + (oil effect)
        predicted_sizes = 110 + (oil_range * 1.5)
        fig = px.line(x=oil_range, y=predicted_sizes, title="Sensitivity of Size to Oil %", labels={'x':'Oil %', 'y':'Size (nm)'})
        st.plotly_chart(fig)

    # Result Table with 6 Parameters
    st.subheader("Final Formulation Audit")
    st.table(pd.DataFrame({
        "Parameter": ["Drug", "Oil", "Surf.", "Co-Surf.", "Water", "S-Mix Ratio"],
        "Value": ["1.0%", f"{st.session_state.oil_p}%", "26.6%", "13.4%", "49.0%", st.session_state.smix_ratio],
        "Optimal Range": ["0.5-2%", "5-15%", "25-40%", "10-20%", "40-60%", "1:1 to 3:1"]
    }))

# --- NAV ---
pg = st.navigation([st.Page(page_discovery, title="1. Search"), st.Page(page_results, title="3. Results")])
pg.run()
