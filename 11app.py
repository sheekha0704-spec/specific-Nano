import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import re
import os
import base64

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Fragments
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v14.0", layout="wide")

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
    .rec-box {
        background: #f8fbff; border: 2px solid #3b82f6; 
        padding: 15px; border-radius: 12px; height: auto; margin-bottom: 10px;
    }
    .summary-table {
        background: #1a202c; color: white; padding: 20px; 
        border-radius: 12px; border-left: 8px solid #f59e0b;
    }
    .summary-table td { padding: 8px; border-bottom: 1px solid #2d3748; }
    .history-item { font-size: 12px; padding: 5px; border-bottom: 1px solid #eee; color: #444; }
    </style>
    """, unsafe_allow_html=True)

# --- PDF GENERATOR FUNCTION ---
def create_pdf(data_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "NanoPredict AI - Formulation Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "1. Chemical Identification", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, val in data_dict['Chemical'].items():
        pdf.cell(200, 8, f"{key}: {val}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "2. Predicted Performance Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, val in data_dict['Results'].items():
        pdf.cell(200, 8, f"{key}: {val}", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"File '{csv_file}' not found.")
        st.stop()
    df = pd.read_csv(csv_file)
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df_train[col] = df_train[col].fillna("Not Specified").astype(str)

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    return df_train, models, le_dict

df, models, le_dict = load_and_prep()

# --- 2. STATE MANAGEMENT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
if 'history' not in st.session_state: st.session_state.history = []
state_keys = ['drug', 'oil', 'aq', 'drug_mg', 'oil_p', 'smix_p', 'smix_ratio', 's_final', 'cs_final', 'fgroups', 'water_p']
for key in state_keys:
    if key not in st.session_state: st.session_state[key] = None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict AI")
    nav_options = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Navigation", nav_options, index=nav_options.index(st.session_state.step_val))

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API & Structural Analysis")
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("Select API (Drug)", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
        aq = st.selectbox("Select Aqueous Phase", ["Distilled Water", "Buffer pH 6.8", "Saline"])
        if st.button("Confirm Phase Setup →"):
            st.session_state.drug, st.session_state.oil, st.session_state.aq = drug, oil, aq
            go_to_step("Step 2: Concentrations")
    with c2:
        if HAS_CHEM_LIBS:
            try:
                comp = pcp.get_compounds(drug, 'name')[0]
                mol = Chem.MolFromSmiles(comp.canonical_smiles)
                st.image(Draw.MolToImage(mol, size=(300,300)), caption=f"Chemical Structure: {drug}")
                fgroups = []
                if Fragments.fr_Al_OH(mol) > 0: fgroups.append("Alcohol (-OH)")
                if Fragments.fr_NH2(mol) > 0: fgroups.append("Primary Amine")
                if Fragments.fr_C_O(mol) > 0: fgroups.append("Carbonyl Group")
                if Fragments.fr_COO(mol) > 0: fgroups.append("Ester/Acid")
                st.session_state.fgroups = fgroups
                st.write(f"**Identified Groups:** {', '.join(fgroups) if fgroups else 'Complex'}")
            except: st.warning("Structure unavailable.")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Formulation Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.oil_p = st.number_input("Oil Phase (%)", 5.0, 50.0, 15.0)
        st.session_state.smix_p = st.number_input("Total S-mix (%)", 5.0, 60.0, 30.0)
        st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        if st.button("Screen Components →"): go_to_step("Step 3: AI Screening")
    with c2: st.metric("Water %", f"{st.session_state.water_p}%")

# --- STEP 3: AI SCREENING (COMPACT) ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Suggested Components")
    best_data = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rec-box"><b>Recommended Surfactants</b>', unsafe_allow_html=True)
        for s in best_data['Surfactant'].unique()[:5]: st.write(f"✅ {s}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rec-box"><b>Suggested Co-Surfactants</b>', unsafe_allow_html=True)
        for cs in best_data['Co-surfactant'].unique()[:5]: st.write(f"🔗 {cs}")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Proceed to Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Selection")
    c1, c2 = st.columns(2)
    with c1:
        s_final = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        cs_final = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        if st.button("Generate Prediction →"):
            st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
            go_to_step("Step 5: Results")
    with c2:
        fg_text = ", ".join(st.session_state.fgroups) if st.session_state.fgroups else "None"
        st.markdown(f'<div class="summary-table"><b>Summary</b><br>Drug: {st.session_state.drug}<br>Groups: {fg_text}<br>Oil: {st.session_state.oil_p}%</div>', unsafe_allow_html=True)

# --- STEP 5: RESULTS & PDF ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: Performance Results")
    inputs = [[le_dict['Drug_Name'].transform([st.session_state.drug])[0],
               le_dict['Oil_phase'].transform([st.session_state.oil])[0],
               le_dict['Surfactant'].transform([st.session_state.s_final])[0],
               le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
    
    res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
    
    cols = st.columns(3)
    metrics = [("Size", f"{res[0]:.1f} nm"), ("PDI", f"{res[1]:.3f}"), ("EE %", f"{res[4]:.1f} %")]
    for i, (l, v) in enumerate(metrics):
        with cols[i]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    # Visualization
    c1, c2 = st.columns(2)
    with c1:
        mu, sigma = res[0], res[0] * res[1]
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
        st.plotly_chart(px.line(x=x, y=y, title="PDI Size Distribution"), use_container_width=True)
    with c2:
        fig_tern = go.Figure(go.Scatterternary({'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p], 'marker': {'size': 14}}))
        st.plotly_chart(fig_tern, use_container_width=True)

    # PDF Export
    report_data = {
        "Chemical": {"API": st.session_state.drug, "Groups": ", ".join(st.session_state.fgroups) if st.session_state.fgroups else "N/A", "Oil": st.session_state.oil},
        "Results": {"Size": f"{res[0]:.2f} nm", "PDI": f"{res[1]:.3f}", "Encapsulation": f"{res[4]:.2f} %"}
    }
    
    pdf_bytes = create_pdf(report_data)
    st.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name=f"NanoReport_{st.session_state.drug}.pdf", mime="application/pdf")

    if st.button("🔄 Restart"): go_to_step("Step 1: Chemical Setup")
