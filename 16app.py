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
st.set_page_config(page_title="NanoPredict Pro", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-top: 4px solid #007bff; text-align: center; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #007bff; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_resource
def load_and_prep(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Option 1: Already installed CSV
        df = pd.read_csv('nanoemulsion 2.csv')
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Specified").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: 
        df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series([1]*len(df_train))).astype(str).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- SIDEBAR: NEW CSV & HISTORY ---
with st.sidebar:
    st.header("⚙️ Data Management")
    uploaded_file = st.file_uploader("📂 Option 3: Upload New CSV Dataset", type=["csv"])
    df_raw, models, stab_model, le_dict = load_and_prep(uploaded_file)
    
    if "history" not in st.session_state: st.session_state.history = []
    if st.session_state.history:
        st.header("📜 History")
        for h in reversed(st.session_state.history[-5:]):
            st.info(f"{h['drug']} | {h['oil']}")

# --- HELPERS ---
def find_best_match(smiles):
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return None
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    best_match, max_sim = None, 0
    for drug in le_dict['Drug_Name'].classes_:
        if drug in ["Not Specified", "New API"]: continue
        try:
            comp = pcp.get_compounds(drug, 'name')[0]
            target_mol = Chem.MolFromSmiles(comp.canonical_smiles)
            sim = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(target_mol, 2))
            if sim > max_sim: max_sim, best_match = sim, drug
        except: continue
    return best_match

# --- 5-STEP NAVIGATION ---
if 'step' not in st.session_state: st.session_state.step = 1

# STEP 1: COMPOUND SETUP
if st.session_state.step == 1:
    st.header("Step 1: Chemical Setup")
    mode = st.radio("API Entry Path", ["Database Drug", "New API (SMILES)"])
    
    if mode == "Database Drug":
        drug_name = st.selectbox("Select API", sorted(le_dict['Drug_Name'].classes_))
        st.session_state.match_drug = drug_name
        try:
            comp = pcp.get_compounds(drug_name, 'name')[0]
            mol = Chem.MolFromSmiles(comp.canonical_smiles)
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            st.image(Draw.MolToImage(mol, size=(300, 200)))
        except:
            st.session_state.logp, st.session_state.mw = 3.0, 300.0
    else:
        smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            st.session_state.match_drug = find_best_match(smiles)
            st.image(Draw.MolToImage(mol, size=(300, 200)))
            st.success(f"Structural Match: {st.session_state.match_drug}")

    st.session_state.oil = st.selectbox("Oil Phase", sorted(le_dict['Oil_phase'].classes_))
    if st.button("Next: Step 2"):
        st.session_state.step = 2
        st.rerun()

# STEP 2: FORMULATION
elif st.session_state.step == 2:
    st.header("Step 2: Formulation Design")
    
    # Solubility Calculation: LogS = 0.5 - 0.01(MW-25) - 0.6(LogP)
    calc_sol = 10**(0.5 - (0.01 * (st.session_state.mw - 50)) - (0.6 * st.session_state.logp)) * 1000
    st.session_state.sol_limit = np.clip(calc_sol, 1.0, 500.0)
    st.info(f"AI Calculated Solubility: **{st.session_state.sol_limit:.2f} mg/mL**")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.session_state.drug_conc = st.number_input("Drug Conc (mg/mL)", 0.1, 500.0, 2.0)
        st.session_state.oil_p = st.slider("Oil %", 1.0, 50.0, 15.0)
        st.session_state.smix_p = st.slider("Smix %", 5.0, 70.0, 30.0)
    with col_b:
        st.session_state.smix_ratio = st.selectbox("Smix Ratio (S:Co-S)", ["1:1", "1:2", "2:1", "3:1", "4:1"])
        st.write(f"**Water Phase:** {100 - st.session_state.oil_p - st.session_state.smix_p}%")

    if st.button("Next: Step 3"):
        st.session_state.step = 3
        st.rerun()

# STEP 3: SCREENING
elif st.session_state.step == 3:
    st.header("Step 3: Database Screening")
    relevant = df_raw[df_raw['Oil_phase'] == st.session_state.oil]
    st.table(relevant.nlargest(5, 'Encapsulation_Efficiency_clean')[['Surfactant', 'Encapsulation_Efficiency']])
    if st.button("Next: Step 4"):
        st.session_state.step = 4
        st.rerun()

# STEP 4: SELECTION
elif st.session_state.step == 4:
    st.header("Step 4: Final Selection")
    st.session_state.s_final = st.selectbox("Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    if st.button("Predict Results"):
        st.session_state.step = 5
        st.rerun()

# STEP 5: RESULTS
elif st.session_state.step == 5:
    st.header("Step 5: AI Performance Suite")
    
    try:
        # Prevent Encoding Error by using matched drug from database
        proxy_drug = st.session_state.match_drug if st.session_state.match_drug else le_dict['Drug_Name'].classes_[0]
        
        inputs = [[le_dict['Drug_Name'].transform([proxy_drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = {k: models[k].predict(inputs)[0] for k in models}
        stable = stab_model.predict(inputs)[0]

        # 1. SIX OUTPUTS
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><p>Size</p><h3>{res['Size_nm']:.1f} nm</h3></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><p>PDI</p><h3>{res['PDI']:.3f}</h3></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><p>Zeta Potential</p><h3>{res['Zeta_mV']:.1f} mV</h3></div>", unsafe_allow_html=True)
        
        c4, c5, c6 = st.columns(3)
        c4.markdown(f"<div class='metric-card'><p>EE %</p><h3>{res['Encapsulation_Efficiency']:.1f}%</h3></div>", unsafe_allow_html=True)
        c5.markdown(f"<div class='metric-card'><p>Solubility</p><h3>{st.session_state.sol_limit:.1f} mg/mL</h3></div>", unsafe_allow_html=True)
        c6.markdown(f"<div class='metric-card'><p>Stability</p><h3>{'Stable' if stable==1 else 'Unstable'}</h3></div>", unsafe_allow_html=True)

        # 2. PLOTS
        v1, v2 = st.columns(2)
        with v1:
            mu, sigma = res['Size_nm'], res['Size_nm'] * (res['PDI'] + 0.05)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            y = np.exp(-0.5 * ((x - mu)/sigma)**2)
            st.plotly_chart(px.line(x=x, y=y, title="Particle Size Distribution"), use_container_width=True)
        with v2:
            fig = go.Figure(go.Scatterternary({
                'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p],
                'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 14}
            }))
            fig.update_layout(title="Ternary Diagram", ternary={'sum': 100, 'aaxis':{'title':'Oil'}, 'baxis':{'title':'Smix'}})
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.history.append({"drug": proxy_drug, "oil": st.session_state.oil})
    except Exception as e:
        st.error(f"Prediction Error: {e}")

    if st.button("Restart"):
        st.session_state.step = 1
        st.rerun()
