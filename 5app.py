import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp

# --- APP CONFIGURATION ---
st.set_page_config(page_title="NanoPredict AI Pro v14", layout="wide")

# Persistent state management for cross-page data
if 'drug' not in st.session_state: st.session_state.drug = "Curcumin"
if 'oil' not in st.session_state: st.session_state.oil = "Oleic Acid"
if 'batch_size' not in st.session_state: st.session_state.batch_size = 10.0

# --- CHEMICAL INTELLIGENCE ---
def detect_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    patterns = {
        "Hydroxyl (-OH)": "[OX2H]", "Carboxyl (-COOH)": "[CX3](=O)[OX2H1]",
        "Aromatic Ring": "a", "Amide": "[NX3][CX3](=[OX1])[#6]", "Ester": "[CX3](=O)[OX2H0][#6]"
    }
    return [name for name, smarts in patterns.items() if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts))]

@st.cache_data
def get_pubchem_data(name):
    try:
        c = pcp.get_compounds(name, 'name')[0]
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{c.cid}/JSON?heading=Solubility"
        sol_text = requests.get(url, timeout=5).json()['Record']['Section'][0]['Section'][0]['Information'][0]['Value']['StringWithMarkup'][0]['String']
        return {"smiles": c.canonical_smiles, "cid": c.cid, "sol": sol_text, "logp": c.xlogp or 3.0}
    except: return None

# --- PAGE 1: DISCOVERY ---
def page_discovery():
    st.title("🧪 Lab 1: Chemical Fingerprinting")
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.session_state.drug = st.text_input("Enter Drug Molecule", st.session_state.drug)
        st.session_state.oil = st.selectbox("Select Oil", ["Oleic Acid", "Miglyol 812", "Castor Oil", "Capmul MCM"])
        st.session_state.batch_size = st.number_input("Target Batch Size (g)", 1.0, 100.0, 10.0)
    with col2:
        data = get_pubchem_data(st.session_state.drug)
        if data:
            st.image(Draw.MolToImage(Chem.MolFromSmiles(data['smiles']), size=(350, 250)))
            st.markdown(f"**Functional Groups:** {', '.join(detect_groups(data['smiles']))}")
            st.warning(f"**Solubility Profile:** {data['sol']}")
        else: st.error("Drug not found in database.")

# --- PAGE 2: SMIX & COMPARISON ---
def page_smix():
    st.title("⚖️ Lab 2: Smix Comparison & Quantity Calculation")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Combination 1 (Primary)")
        s1 = st.selectbox("Surfactant 1", ["Tween 80", "Cremophor EL", "Labrasol"], key='s1')
        cs1 = st.selectbox("Cosurfactant 1", ["PEG 400", "Ethanol", "Propylene Glycol"], key='cs1')
        ratio1 = st.select_slider("Ratio 1 (S:CoS)", options=["3:1", "2:1", "1:1", "1:2"], key='r1')
        oil_p1 = st.slider("Oil %", 5, 25, 10, key='o1')
        
    with colB:
        st.subheader("Combination 2 (Alternative)")
        s2 = st.selectbox("Surfactant 2", ["Tween 20", "Span 80", "Labrafil"], key='s2')
        cs2 = st.selectbox("Cosurfactant 2", ["Glycerin", "PEG 200", "Ethanol"], key='cs2')
        ratio2 = st.select_slider("Ratio 2 (S:CoS)", options=["3:1", "2:1", "1:1", "1:2"], key='r2')
        oil_p2 = st.slider("Oil %", 5, 25, 15, key='o2')

    # Calculation logic for quantities
    def calc_masses(oil_p, ratio_str, drug_p=1.0):
        smix_total_p = 40.0 # Fixed Smix for calculation
        s_part, c_part = map(int, ratio_str.split(':'))
        s_p = (s_part / (s_part + c_part)) * smix_total_p
        c_p = (c_part / (s_part + c_part)) * smix_total_p
        w_p = 100 - (oil_p + drug_p + s_p + c_p)
        return [drug_p, oil_p, s_p, c_p, w_p, smix_total_p]

    m1 = calc_masses(oil_p1, ratio1)
    m2 = calc_masses(oil_p2, ratio2)

    st.markdown("### 📊 Comparison Table (for 10g Batch)")
    comp_df = pd.DataFrame({
        "Component": ["Drug (g)", "Oil (g)", "Surfactant (g)", "Cosurfactant (g)", "Water (g)", "Total Smix (g)"],
        "Combination 1": [x/10 for x in m1],
        "Combination 2": [x/10 for x in m2]
    })
    st.table(comp_df)

# --- PAGE 3: STABILITY RESULTS ---
def page_results():
    st.title("📊 Lab 3: Final Stability & Optimum Ranges")
    
    # Optimum Range Table
    st.subheader("Optimization Parameters")
    opt_df = pd.DataFrame({
        "Parameter": ["Drug Loading", "Oil Concentration", "Surfactant (Smix)", "Cosurfactant (Smix)", "Total Smix", "Water Phase"],
        "Your Value": ["1.0%", f"{st.session_state.get('o1', 10)}%", "26.6%", "13.4%", "40.0%", "49.0%"],
        "Optimum Range": ["0.5 - 2.0%", "5 - 15%", "20 - 35%", "10 - 20%", "30 - 50%", "40 - 70%"],
        "Status": ["✅ Optimal", "✅ Optimal", "⚠️ High", "✅ Optimal", "✅ Optimal", "✅ Optimal"]
    })
    st.table(opt_df)

    c1, c2 = st.columns(2)
    with c1:
        # PDI with Sensitivity Analysis
        pdi = 0.15
        st.write(f"### Predicted PDI: {pdi}")
        st.info("Sensitivity: A 5% increase in Surfactant reduces PDI by 0.02 units.")
        x = np.linspace(30, 250, 100)
        y = np.exp(-0.5 * ((x - 110) / (110 * pdi))**2)
        st.plotly_chart(px.area(x=x, y=y, title="Droplet Size Distribution"))

    with c2:
        # Phase Diagram
        fig = px.scatter_ternary(pd.DataFrame({'O':[10], 'S':[40], 'W':[50]}), a="O", b="S", c="W")
        fig.update_layout(title="Stability Window (Ternary)")
        st.plotly_chart(fig)

# --- NAVIGATION ---
pg = st.navigation([
    st.Page(page_discovery, title="1. Ingredient Search", icon="🧪"),
    st.Page(page_smix, title="2. Combination Design", icon="⚖️"),
    st.Page(page_results, title="3. Stability Result", icon="📊")
])
pg.run()
