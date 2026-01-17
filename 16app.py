import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem
    import pubchempy as pcp
except ImportError:
    st.error("Missing RDKit or PubChemPy. Please install them."); st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v23.0 - Global Unknown Predictor", layout="wide")

# --- DATABASE & PARAMETERS ---
OIL_HSP = {
    "Capryol 90": [15.8, 8.2, 10.4], "Oleic Acid": [16.4, 3.3, 5.5],
    "Castor Oil": [16.1, 5.2, 9.9], "Olive Oil": [16.5, 3.1, 4.8],
    "Labrafac": [15.7, 5.6, 8.0], "Isopropyl Myristate": [16.2, 3.9, 3.7]
}

# --- 1. AI ENGINE (REBUILT FOR UNKNOWN COMPOUNDS) ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error("Missing 'nanoemulsion 2.csv'"); st.stop()
    df = pd.read_csv(csv_file)
    
    # Clean Categorical Data to prevent Sort/TypeError crashes
    cat_cols = ['Surfactant', 'Co-surfactant', 'Oil_phase', 'Drug_Name']
    for col in cat_cols:
        df[col] = df[col].fillna("None").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    # IMPORTANT: Convert existing database drugs to LogP/MW so model learns patterns
    @st.cache_data(ttl=3600)
    def get_train_props(name):
        try:
            # Fallback data if PubChem is busy
            if name == "None": return 3.0, 300.0
            comp = pcp.get_compounds(name, 'name')[0]
            mol = Chem.MolFromSmiles(comp.canonical_smiles)
            return Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
        except: return 3.5, 350.0 # Standard fallback

    unique_drugs = df['Drug_Name'].unique()
    drug_map = {d: get_train_props(d) for d in unique_drugs}
    df['Drug_LogP'] = df['Drug_Name'].map(lambda x: drug_map[x][0])
    df['Drug_MW'] = df['Drug_Name'].map(lambda x: drug_map[x][1])

    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    le_dict = {}
    for col in ['Oil_phase', 'Surfactant', 'Co-surfactant']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    # Features: LogP, MW, and Ingredient Encodings
    X = df_train[['Drug_LogP', 'Drug_MW', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df_raw, models, stab_model, le_dict = load_and_prep()

# --- NAVIGATION ---
if 'step' not in st.session_state: st.session_state.step = 1

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step == 1:
    st.header("Step 1: Unknown Compound Setup")
    c1, c2 = st.columns(2)
    with c1:
        smiles = st.text_input("Enter SMILES for Unknown Drug", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            st.session_state.smiles = smiles
            st.success(f"LogP: {st.session_state.logp:.2f} | MW: {st.session_state.mw:.2f}")
            st.image(Draw.MolToImage(mol, size=(300, 200)), caption="Molecular Structure")
            
            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(list(OIL_HSP.keys())))
            if st.button("Proceed to Concentrations →"): st.session_state.step = 2; st.rerun()
        else:
            st.error("Please enter a valid SMILES string.")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step == 2:
    st.header("Step 2: Component Ratios")
    st.session_state.oil_p = st.number_input("Oil Phase %", 5.0, 40.0, 15.0)
    st.session_state.smix_p = st.number_input("S-mix %", 10.0, 60.0, 30.0)
    if st.button("Proceed to Screening →"): st.session_state.step = 3; st.rerun()

# --- STEP 3: SCREENING ---
elif st.session_state.step == 3:
    st.header("Step 3: AI Ingredient Screening")
    best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values('Encapsulation_Efficiency_clean', ascending=False)
    st.write("Top surfactants based on your oil selection:")
    st.dataframe(best_data[['Surfactant', 'Encapsulation_Efficiency']].drop_duplicates().head(5))
    if st.button("Proceed to Final Selection →"): st.session_state.step = 4; st.rerun()

# --- STEP 4: SELECTION ---
elif st.session_state.step == 4:
    st.header("Step 4: Final Formulation Selection")
    # FIX: sorted([str(x)...]) prevents TypeError during sorting
    surfactants = sorted([str(s) for s in le_dict['Surfactant'].classes_])
    co_surfactants = sorted([str(cs) for cs in le_dict['Co-surfactant'].classes_])
    
    st.session_state.s_final = st.selectbox("Select Surfactant", surfactants)
    st.session_state.cs_final = st.selectbox("Select Co-Surfactant", co_surfactants)
    if st.button("Generate AI Predictions →"): st.session_state.step = 5; st.rerun()

# --- STEP 5: RESULTS ---
elif st.session_state.step == 5:
    st.header("Step 5: Final Performance Results")
    try:
        # Prediction using numerical features (fixes unseen label error)
        inputs = [[
            st.session_state.logp, 
            st.session_state.mw,
            le_dict['Oil_phase'].transform([st.session_state.oil])[0],
            le_dict['Surfactant'].transform([st.session_state.s_final])[0],
            le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]
        ]]
        
        preds = {k: models[k].predict(inputs)[0] for k in models}
        
        cols = st.columns(4)
        cols[0].metric("Size (nm)", f"{preds['Size_nm']:.1f}")
        cols[1].metric("PDI", f"{preds['PDI']:.3f}")
        cols[2].metric("Zeta (mV)", f"{preds['Zeta_mV']:.1f}")
        cols[3].metric("EE %", f"{preds['Encapsulation_Efficiency']:.1f}")

        # Stability Phase Diagram
        grid = 10; o_rng = np.linspace(5, 40, grid); s_rng = np.linspace(10, 50, grid); z = np.zeros((grid, grid))
        for i, o in enumerate(o_rng):
            for j, s in enumerate(s_rng):
                z[i,j] = stab_model.predict(inputs)[0]
        st.plotly_chart(px.imshow(z, x=s_rng, y=o_rng, labels=dict(x="Smix %", y="Oil %"), title="Stability Map"))

    except Exception as e:
        st.error(f"Prediction Error: {e}")
    
    if st.button("Start New Formulation"): st.session_state.step = 1; st.rerun()
