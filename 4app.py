import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
try:
    import pubchempy as pcp
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError: pass

# --- GLOBAL CONFIGURATION ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")

# Initialize Session State for cross-page persistence
if 'drug_name' not in st.session_state: st.session_state.drug_name = "Curcumin"
if 'selected_oil' not in st.session_state: st.session_state.selected_oil = "Oleic Acid"
if 'selected_surf' not in st.session_state: st.session_state.selected_surf = "Tween 80"

# Data Constants
OILS = ["Oleic Acid", "Caprylic triglyceride", "Castor Oil", "Miglyol 812", "Soybean Oil", "Coconut Oil"]
SURFACTANTS = ["Tween 80", "Span 80", "Cremophor EL", "Solutol HS15", "Labrasol", "Lecithin"]

# --- HELPER FUNCTIONS ---
@st.cache_data
def get_chem_data(name):
    try:
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            mol = Chem.MolFromSmiles(c.canonical_smiles)
            return {"mw": c.molecular_weight, "logp": c.xlogp or 2.0, "img": Draw.MolToImage(mol, size=(300,300))}
    except: return None

@st.cache_data
def get_research_links(drug):
    links = []
    try:
        url = f"https://scholar.google.com/scholar?q={drug}+nanoemulsion+2023..2026"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for item in soup.select('.gs_r.gs_or')[:5]:
            title = item.select_one('.gs_rt').text
            link = item.select_one('.gs_rt a')['href']
            links.append({"title": title, "url": link})
    except: pass
    return links

# --- PAGE DEFINITIONS ---

def page_1_discovery():
    st.title("💊 Page 1: Drug & Ingredient Discovery")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.session_state.drug_name = st.text_input("Search Drug/Phytochemical", st.session_state.drug_name)
        st.session_state.selected_oil = st.selectbox("Select Target Oil Phase", OILS)
        
    with col2:
        data = get_chem_data(st.session_state.drug_name)
        if data:
            st.image(data['img'], width=300)
            st.success(f"**{st.session_state.drug_name}** found in database.")
            st.info(f"MW: {data['mw']} | LogP: {data['logp']}")
        else:
            st.warning("Structure not found. Using general prediction model.")

def page_2_literature():
    st.title("📚 Page 2: Recommended Research (2023-2026)")
    st.write(f"Showing recent nanoemulsion publications for: **{st.session_state.drug_name}**")
    
    articles = get_research_links(st.session_state.drug_name)
    if articles:
        for art in articles:
            st.markdown(f"🔗 [{art['title']}]({art['url']})")
    else:
        st.write("No specific recent articles found. Try a different synonym.")

def page_3_prediction():
    st.title("📊 Page 3: AI Formulation & Ternary Analysis")
    
    # 1. Custom Ternary Plot
    st.subheader("Interactive Ternary Phase Diagram")
    oil_val = st.slider("Oil %", 5, 50, 20)
    surf_val = st.slider("Surfactant %", 10, 80, 40)
    water_val = 100 - (oil_val + surf_val)
    
    fig_ternary = px.scatter_ternary(
        pd.DataFrame({'A': [oil_val], 'B': [surf_val], 'C': [water_val]}),
        a="A", b="B", c="C", labels={'A':'Oil', 'B':'Surfactant', 'C':'Water'}
    )
    st.plotly_chart(fig_ternary)

    # 2. PDI Graph
    st.subheader("Polydispersity Index (PDI) Distribution")
    # AI Simulation for PDI
    size = 150 # Placeholder for model prediction
    pdi = 0.22 # Placeholder for model prediction
    x = np.linspace(size-100, size+100, 100)
    y = np.exp(-0.5 * ((x - size) / (size * pdi))**2)
    
    fig_pdi = px.line(x=x, y=y, title=f"Predicted Size Distribution (PDI: {pdi})")
    st.plotly_chart(fig_pdi)

# --- NAVIGATION LOGIC ---
pg = st.navigation([
    st.Page(page_1_discovery, title="Ingredient Discovery", icon="🧪"),
    st.Page(page_2_literature, title="Research Articles", icon="📖"),
    st.Page(page_3_prediction, title="AI Prediction", icon="📈")
])
pg.run()
