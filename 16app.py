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
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE (Handles Custom CSV & AI Training) ---
@st.cache_resource
def load_and_prep(file):
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Fallback to your local file
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
    
    # Stability Model
    df_train['is_stable'] = df_train.get('Stability', pd.Series([1]*len(df_train))).astype(str).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- SIDEBAR: CSV UPLOAD & HISTORY ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload your own CSV file", type=["csv"])
    
    # Load data based on sidebar upload
    df_raw, models, stab_model, le_dict = load_and_prep(uploaded_file)
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        st.header("📜 Formulation History")
        for i, entry in enumerate(reversed(st.session_state.history[-5:])):
            st.info(f"{entry['drug']} in {entry['oil']}")

# --- STRUCTURAL MATCHING (Solves "Unseen Labels" Error) ---
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
    smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.session_state.logp = Descriptors.MolLogP(mol)
        st.session_state.match_drug = find_best_match(smiles)
        st.image(Draw.MolToImage(mol, size=(300, 200)))
        st.session_state.oil = st.selectbox("Select Oil", sorted(le_dict['Oil_phase'].classes_))
        if st.button("Next: Step 2"): 
            st.session_state.step = 2
            st.rerun()

# STEP 2: AUTO-SOLUBILITY & CONCENTRATION
elif st.session_state.step == 2:
    st.header("Step 2: AI Predicted Solubility & Loading")
    # AI Logic: Solubility estimated based on LogP
    pred_sol = np.maximum(5.0, (450 / (st.session_state.logp + 1))) 
    st.session_state.sol_limit = st.slider("Predicted Solubility (mg/mL)", 1.0, 500.0, float(pred_sol))
    
    st.session_state.drug_conc = st.number_input("Select Drug Concentration (mg/mL)", 0.1, st.session_state.sol_limit, 2.0)
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    if st.button("Next: Step 3"): 
        st.session_state.step = 3
        st.rerun()

# STEP 3: SCREENING
elif st.session_state.step == 3:
    st.header("Step 3: Screening")
    st.write("Top surfactants in database for selected oil:")
    relevant = df_raw[df_raw['Oil_phase'] == st.session_state.oil]
    st.table(relevant.nlargest(5, 'Encapsulation_Efficiency_clean')[['Surfactant', 'Encapsulation_Efficiency']])
    if st.button("Next: Step 4"): 
        st.session_state.step = 4
        st.rerun()

# STEP 4: FINAL SELECTION
elif st.session_state.step == 4:
    st.header("Step 4: Final Ingredients")
    st.session_state.s_final = st.selectbox("Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    if st.button("Predict Results"): 
        st.session_state.step = 5
        st.rerun()

# STEP 5: 6 MAJOR OUTPUTS & PLOTS
elif st.session_state.step == 5:
    st.header("Step 5: Results & Visualizations")
    
    # Use the matched drug to prevent encoding errors
    proxy_drug = st.session_state.match_drug if st.session_state.match_drug else le_dict['Drug_Name'].classes_[0]
    
    try:
        inputs = [[le_dict['Drug_Name'].transform([proxy_drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = {k: models[k].predict(inputs)[0] for k in models}
        stability = stab_model.predict(inputs)[0]

        # 1. SIX MAJOR OUTPUTS
        c1, c2, c3 = st.columns(3)
        c1.metric("Size (nm)", f"{res['Size_nm']:.1f}")
        c2.metric("PDI", f"{res['PDI']:.3f}")
        c3.metric("Zeta (mV)", f"{res['Zeta_mV']:.1f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("EE %", f"{res['Encapsulation_Efficiency']:.1f}%")
        c5.metric("Solubility", f"{st.session_state.sol_limit:.1f} mg/mL")
        c6.metric("Stability", "Stable" if stability == 1 else "Unstable")

        # 2. VISUALIZATIONS
        col_a, col_b = st.columns(2)
        
        # Distribution Curve
        x = np.linspace(res['Size_nm']-60, res['Size_nm']+60, 100)
        y = np.exp(-0.5 * ((x - res['Size_nm']) / (res['Size_nm'] * (res['PDI']+0.05)))**2)
        col_a.plotly_chart(px.line(x=x, y=y, title="Size Distribution Curve"), use_container_width=True)
        
        # Ternary Diagram
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.oil_p],
            'b': [st.session_state.smix_p],
            'c': [100-st.session_state.oil_p-st.session_state.smix_p],
            'marker': {'color': 'blue', 'size': 14}
        }))
        fig.update_layout(title="Ternary Phase Diagram", ternary={'sum': 100, 'aaxis':{'title':'Oil'}, 'baxis':{'title':'Smix'}})
        col_b.plotly_chart(fig, use_container_width=True)

        if not any(d['drug'] == proxy_drug for d in st.session_state.history):
            st.session_state.history.append({"drug": proxy_drug, "oil": st.session_state.oil})

    except Exception as e:
        st.error(f"Error: {e}")

    if st.button("New Prediction"): 
        st.session_state.step = 1
        st.rerun()
