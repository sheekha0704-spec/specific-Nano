import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, AllChem, DataStructs
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict Pro v20.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; margin-bottom: 20px; }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    .summary-table { background: #1a202c; color: white; padding: 20px; border-radius: 12px; border-left: 8px solid #f59e0b; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & AI ENGINE ---
@st.cache_resource
def load_and_prep(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv('nanoemulsion 2.csv') if os.path.exists('nanoemulsion 2.csv') else pd.DataFrame()
    
    if df.empty:
        st.error("No data found. Please upload a CSV in the sidebar.")
        st.stop()

    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Specified").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).astype(str).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- SIDEBAR & HISTORY ---
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
    df_raw, models, stab_model, le_dict = load_and_prep(uploaded_file)
    
    if "history" not in st.session_state: st.session_state.history = []
    if st.session_state.history:
        st.write("### History")
        for h in reversed(st.session_state.history[-3:]):
            st.caption(f"Recent: {h['drug']} in {h['oil']}")

# --- NAVIGATION STATE ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"

def go_to(step_name):
    st.session_state.step_val = step_name
    st.rerun()

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API Analysis & Oil Selection")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.session_state.smiles = smiles
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            
            # Use Morgan Fingerprints to find similar drug in database
            query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            best_match, max_sim = le_dict['Drug_Name'].classes_[0], -1
            
            for d in le_dict['Drug_Name'].classes_:
                if d == "Not Specified": continue
                try:
                    target_comp = pcp.get_compounds(d, 'name')[0]
                    target_mol = Chem.MolFromSmiles(target_comp.canonical_smiles)
                    sim = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(target_mol, 2))
                    if sim > max_sim: max_sim, best_match = sim, d
                except: continue
            
            st.session_state.match_drug = best_match
            st.image(Draw.MolToImage(mol, size=(300, 200)))
            st.info(f"Similarity: {best_match} ({max_sim*100:.1f}%)")
            
            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
            if st.button("Confirm Step 1"): go_to("Step 2: Concentrations")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Predicted Solubility & Loading")
    # AI Logic: Predict solubility based on LogP
    auto_sol = np.clip(500 / (st.session_state.logp + 1), 5, 500)
    
    st.session_state.sol_limit = st.slider("Predicted Solubility (mg/mL)", 1.0, 500.0, float(auto_sol))
    st.session_state.drug_conc = st.number_input("Desired Loading (mg/mL)", 0.1, 500.0, 2.0)
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    
    if st.button("Confirm Step 2"): go_to("Step 3: AI Screening")

# --- STEP 3: SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: AI Component Screening")
    best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    
    st.write("Top surfactants for this oil based on database:")
    st.dataframe(best_data[['Surfactant', 'Encapsulation_Efficiency']].drop_duplicates().head(5))
    
    if st.button("Confirm Step 3"): go_to("Step 4: Selection")

# --- STEP 4: SELECTION ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Formulation Selection")
    st.session_state.s_final = st.selectbox("Final Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Final Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    
    if st.button("Generate AI Results"): go_to("Step 5: Results")

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: AI Performance Suite")
    
    # Map back to known labels for the AI
    inputs = [[le_dict['Drug_Name'].transform([st.session_state.match_drug])[0],
               le_dict['Oil_phase'].transform([st.session_state.oil])[0],
               le_dict['Surfactant'].transform([st.session_state.s_final])[0],
               le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
    
    res = {col: models[col].predict(inputs)[0] for col in models}
    is_stable = stab_model.predict(inputs)[0]

    # THE 6 MAJOR OUTPUTS
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE %</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)
    c5.markdown(f"<div class='metric-card'><div class='m-label'>Solubility</div><div class='m-value'>{st.session_state.sol_limit:.1f}</div></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='metric-card'><div class='m-label'>Stability</div><div class='m-value'>{'Stable' if is_stable==1 else 'Unstable'}</div></div>", unsafe_allow_html=True)

    # VISUALIZATIONS
    t1, t2 = st.tabs(["Distribution Curve", "Ternary Phase Diagram"])
    with t1:
        # Gaussian distribution based on PDI
        mu, sigma = res['Size_nm'], res['Size_nm'] * (res['PDI'] + 0.05)
        x_vals = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y_vals = np.exp(-0.5 * ((x_vals - mu)/sigma)**2)
        st.plotly_chart(px.line(x=x_vals, y=y_vals, title="Particle Size Distribution"), use_container_width=True)
        

[Image of a particle size distribution curve]


    with t2:
        water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'size': 18, 'color': 'green'}}))
        fig.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil%'}, 'baxis': {'title': 'Smix%'}, 'caxis': {'title': 'Water%'}})
        st.plotly_chart(fig, use_container_width=True)
        

    # Save to history
    st.session_state.history.append({"drug": st.session_state.match_drug, "oil": st.session_state.oil})
    
    if st.button("New Formulation"): go_to("Step 1: Chemical Setup")
