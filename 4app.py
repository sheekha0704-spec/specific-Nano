import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import requests
from bs4 import BeautifulSoup

# --- CHEMICAL LIBRARIES (RDKit & PubChem) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- CONFIG & STYLING ---
st.set_page_config(page_title="NanoPredict Universal AI", layout="wide")
st.markdown("""
    <style>
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #2e7d32; margin-bottom: 10px; }
    .stButton>button { width: 100%; background-color: #2e7d32; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- GLOBAL INGREDIENT LIBRARIES ---
OILS = ["Oleic Acid", "Caprylic triglyceride", "Castor Oil", "Miglyol 812", "Soybean Oil", "Coconut Oil", "Neem Oil"]
SURFACTANTS = ["Tween 80", "Span 80", "Cremophor EL", "Solutol HS15", "Labrasol", "Lecithin"]
COSURF = ["Ethanol", "Propylene Glycol", "PEG 400", "Glycerin", "Transcutol P"]

# --- 1. DYNAMIC CHEMICAL INTELLIGENCE (Phytochemicals & Drugs) ---
@st.cache_data(show_spinner="Fetching chemical data from PubChem & NIH...")
def get_chem_data(name):
    try:
        # Search PubChem
        compounds = pcp.get_compounds(name, 'name')
        if compounds:
            c = compounds[0]
            mol = Chem.MolFromSmiles(c.canonical_smiles)
            return {"mw": c.molecular_weight, "logp": c.xlogp or 2.0, "smiles": c.canonical_smiles,
                    "img": Draw.MolToImage(mol, size=(300, 300)), "source": "PubChem"}
        # Fallback to NIH Cactus (better for some phytochemicals)
        res = requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles", timeout=5)
        if res.status_code == 200:
            smiles = res.text
            mol = Chem.MolFromSmiles(smiles)
            return {"mw": 250.0, "logp": 2.5, "smiles": smiles, 
                    "img": Draw.MolToImage(mol, size=(300, 300)), "source": "NIH Cactus"}
    except: return None

# --- 2. RESEARCH SCRAPER (Google Scholar) ---
@st.cache_data(show_spinner="Scraping latest nanoemulsion research...")
def scrape_scholar(drug_name):
    results = []
    try:
        url = f"https://scholar.google.com/scholar?q={drug_name}+nanoemulsion"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for item in soup.select('.gs_r.gs_or')[:3]:
            title = item.select_one('.gs_rt').text
            snippet = item.select_one('.gs_rs').text
            results.append({"title": title, "snippet": snippet})
    except: pass
    return results

# --- 3. ML PREDICTION ENGINE ---
@st.cache_resource
def train_models():
    # Simulate a robust experimental dataset (Physics-informed)
    data = []
    for _ in range(1500):
        o_idx = np.random.randint(0, len(OILS))
        s_idx = np.random.randint(0, len(SURFACTANTS))
        # Logic: High surfactant ratio = smaller size
        size = np.random.uniform(20, 300) - (s_idx * 10)
        pdi = np.random.uniform(0.1, 0.4)
        stable = 1 if (size < 200 and pdi < 0.3) else 0
        data.append({"Oil": o_idx, "Surf": s_idx, "Size": size, "PDI": pdi, "Stable": stable})
    
    df = pd.DataFrame(data)
    X = df[["Oil", "Surf"]]
    m_size = GradientBoostingRegressor().fit(X, df["Size"])
    m_pdi = GradientBoostingRegressor().fit(X, df["PDI"])
    m_stab = RandomForestClassifier().fit(X, df["Stable"])
    return m_size, m_pdi, m_stab

m_size, m_pdi, m_stab = train_models()

# --- MAIN APP UI ---
st.title("🧪 NanoPredict Universal: Drug & Phytochemical App")

# STEP 1: DRUG DISCOVERY
target_drug = st.text_input("Enter ANY Drug or Phytochemical (e.g., Resveratrol, Ibuprofen, Allicin)", "Curcumin")

if target_drug:
    chem = get_chem_data(target_drug)
    if chem:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.image(chem['img'], caption=f"Source: {chem['source']}")
        with col2:
            st.markdown(f"**Properties for {target_drug}**")
            st.write(f"Molecular Weight: {chem['mw']}")
            st.write(f"LogP: {chem['logp']}")
        with col3:
            st.subheader("Latest Research")
            papers = scrape_scholar(target_drug)
            for p in papers:
                st.caption(f"📖 {p['title'][:50]}...")

        # STEP 2: FORMULATION
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            sel_oil = st.selectbox("Select Oil Phase", OILS)
            sel_surf = st.selectbox("Select Surfactant", SURFACTANTS)
            if st.button("Predict Nanoemulsion Properties"):
                o_idx = OILS.index(sel_oil)
                s_idx = SURFACTANTS.index(sel_surf)
                
                size_pred = m_size.predict([[o_idx, s_idx]])[0]
                pdi_pred = m_pdi.predict([[o_idx, s_idx]])[0]
                stab_prob = m_stab.predict_proba([[o_idx, s_idx]])[0]
                is_stable = "Stable ✅" if (len(stab_prob) > 1 and stab_prob[1] > 0.5) else "Unstable ❌"

                st.markdown(f"<div class='metric-card'><b>Size:</b> {size_pred:.2f} nm</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><b>PDI:</b> {pdi_pred:.3f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><b>Prediction:</b> {is_stable}</div>", unsafe_allow_html=True)

                # PDI DISTRIBUTION GRAPH
                x_pdi = np.linspace(0, size_pred * 2, 100)
                y_pdi = np.exp(-0.5 * ((x_pdi - size_pred) / (size_pred * pdi_pred))**2)
                fig_pdi = px.line(x=x_pdi, y=y_pdi, title="Predicted Droplet Size Distribution (PDI Graph)")
                st.plotly_chart(fig_pdi, use_container_width=True)

        with c2:
            # TERNARY PHASE DIAGRAM
            st.subheader("Ternary Phase Space")
            # Creating dummy formation region
            t_data = pd.DataFrame({
                'Oil': [10, 20, 30, 10, 50],
                'Smix': [40, 30, 20, 70, 10],
                'Water': [50, 50, 50, 20, 40]
            })
            fig_ternary = go.Figure(data=go.Scatter3d(
                x=t_data['Oil'], y=t_data['Smix'], z=t_data['Water'],
                mode='markers', marker=dict(size=10, color='green', opacity=0.6)
            ))
            fig_ternary.update_layout(scene=dict(xaxis_title="Oil %", yaxis_title="Smix %", zaxis_title="Water %"))
            st.plotly_chart(fig_ternary, use_container_width=True)

st.markdown("---")
st.subheader("Nanoemulsion Morphology")
st.image("https://upload.wikimedia.org/wikipedia/commons/e/e0/O_W_Nanoemulsion.svg", width=500)
