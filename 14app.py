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
    from rdkit.Chem import Draw, Descriptors, Fragments, AllChem, DataStructs
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v17.0 - Solubility Suite", layout="wide")

# --- DATABASE & PARAMETERS ---
# Hansen Solubility Parameters (dD, dP, dH) for Oils
OIL_HSP = {
    "Capryol 90": [15.8, 8.2, 10.4],
    "Oleic Acid": [16.4, 3.3, 5.5],
    "Castor Oil": [16.1, 5.2, 9.9],
    "Olive Oil": [16.5, 3.1, 4.8],
    "Labrafac": [15.7, 5.6, 8.0],
    "Isopropyl Myristate": [16.2, 3.9, 3.7]
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; margin-bottom: 20px; }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    .algo-box { background: #f0f7ff; border: 1px dashed #007bff; padding: 15px; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & SIMILARITY ENGINE ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error("Database file 'nanoemulsion 2.csv' not found."); st.stop()
    df = pd.read_csv(csv_file)
    
    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col].astype(str))
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    # Stability Model
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df_raw, models, stab_model, le_dict = load_and_prep()

def get_hsp_from_smiles(smiles):
    """Estimate HSP based on molecular descriptors (Heuristic/Group Contribution)"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [0, 0, 0]
    # Simple heuristic for demonstration:
    dD = Descriptors.MolWt(mol) / 20.0
    dP = Fragments.fr_Ar_OH(mol) * 4.0 + Fragments.fr_C_O(mol) * 3.0 + 2.0
    dH = Fragments.fr_Al_OH(mol) * 8.0 + Fragments.fr_NH2(mol) * 5.0 + 1.0
    return [dD, dP, dH]

def find_best_match(smiles):
    """Find most structurally similar drug in database"""
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return None, 0
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    best_match, max_sim = None, 0
    for drug in df_raw['Drug_Name'].unique():
        try:
            target_comp = pcp.get_compounds(drug, 'name')[0]
            target_mol = Chem.MolFromSmiles(target_comp.canonical_smiles)
            target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)
            sim = DataStructs.TanimotoSimilarity(query_fp, target_fp)
            if sim > max_sim: max_sim, best_match = sim, drug
        except: continue
    return best_match, max_sim

# --- 2. STATE MANAGEMENT ---
if 'step' not in st.session_state: st.session_state.step = "Step 1: Chemical Setup"
keys = ['drug', 'oil', 'smiles', 'sol_limit', 'oil_p', 'smix_p', 's_final', 'cs_final', 'match_drug']
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step == "Step 1: Chemical Setup":
    st.header("Step 1: Drug Structure & Solubility Parameter Analysis")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        mode = st.radio("Input Method", ["Database Drug", "New Proprietary Drug (SMILES)"])
        if mode == "Database Drug":
            drug_name = st.selectbox("Select API", sorted(df_raw['Drug_Name'].unique()))
            comp = pcp.get_compounds(drug_name, 'name')[0]
            smiles = comp.canonical_smiles
        else:
            smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
            drug_name = "New Drug X"
        
        if st.button("Analyze Affinity →"):
            st.session_state.drug, st.session_state.smiles = drug_name, smiles
            st.rerun()

    with c2:
        if st.session_state.smiles:
            mol = Chem.MolFromSmiles(st.session_state.smiles)
            if mol:
                st.image(Draw.MolToImage(mol, size=(400,300)), caption=st.session_state.drug)
                drug_hsp = get_hsp_from_smiles(st.session_state.smiles)
                
                # Solubility Heatmap Logic
                oil_names, distances = [], []
                for name, hsp in OIL_HSP.items():
                    # HSP Distance: Ra = sqrt(4*(dD1-dD2)^2 + (dP1-dP2)^2 + (dH1-dH2)^2)
                    dist = np.sqrt(4*(drug_hsp[0]-hsp[0])**2 + (drug_hsp[1]-hsp[1])**2 + (drug_hsp[2]-hsp[2])**2)
                    oil_names.append(name); distances.append(dist)
                
                h_df = pd.DataFrame({"Oil Phase": oil_names, "HSP Distance": distances}).sort_values("HSP Distance")
                fig = px.bar(h_df, x="HSP Distance", y="Oil Phase", orientation='h', 
                             title="Carrier Affinity (Lower is Better)", color="HSP Distance", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)
                
                match, score = find_best_match(st.session_state.smiles)
                st.session_state.match_drug = match
                st.info(f"Structural Match: **{match}** ({score*100:.1f}% similarity). Model will use this for performance baseline.")
                
                st.session_state.oil = h_df.iloc[0]["Oil Phase"]
                if st.button("Confirm Best Oil & Proceed →"):
                    st.session_state.step = "Step 2: Concentrations"
                    st.rerun()

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step == "Step 2: Concentrations":
    st.header("Step 2: Solubility Limits & Loading")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sol_limit = st.number_input(f"Measured Solubility of Drug in {st.session_state.oil} (mg/mL)", 0.1, 500.0, 20.0)
        st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
        st.session_state.smix_p = st.slider("Total S-mix %", 10, 60, 30)
        if st.button("Calculate Design Space →"):
            st.session_state.step = "Step 3: AI Screening"
            st.rerun()
    with c2:
        max_load = st.session_state.sol_limit * (st.session_state.oil_p / 100)
        st.metric("Theoretical Max Drug Loading", f"{max_load:.2f} mg/mL")
        st.markdown("<div class='algo-box'><b>Note:</b> Exceeding this limit leads to drug precipitation and emulsion instability.</div>", unsafe_allow_html=True)

# --- STEP 3: SCREENING & STEP 4: SELECTION (Simplified for integration) ---
elif st.session_state.step in ["Step 3: AI Screening", "Step 4: Selection"]:
    st.header(st.session_state.step)
    s_final = st.selectbox("Select Surfactant", sorted(df_raw['Surfactant'].unique()))
    cs_final = st.selectbox("Select Co-Surfactant", sorted(df_raw['Co-surfactant'].unique()))
    if st.button("Generate Final Prediction →"):
        st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
        st.session_state.step = "Step 5: Results"
        st.rerun()

# --- STEP 5: RESULTS ---
elif st.session_state.step == "Step 5: Results":
    st.header("Step 5: Final AI Performance & Phase Distribution")
    
    # Use the 'matched' drug for the AI prediction if it's a new drug
    target_drug = st.session_state.match_drug if st.session_state.match_drug else st.session_state.drug
    
    inputs = [[le_dict['Drug_Name'].transform([target_drug])[0],
               le_dict['Oil_phase'].transform([st.session_state.oil])[0],
               le_dict['Surfactant'].transform([st.session_state.s_final])[0],
               le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
    
    res = {col: models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Droplet Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta Potential</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE %</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)

    t1, t2 = st.tabs(["Ternary Phase Diagram", "Stability Heatmap"])
    
    with t1:
        st.subheader("Formulation Design Space")
        water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        fig_tern = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p],
            'marker': {'color': '#28a745', 'size': 18, 'symbol': 'diamond', 'line': {'width': 2, 'color': 'black'}}
        }))
        fig_tern.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}})
        st.plotly_chart(fig_tern, use_container_width=True)

    with t2:
        grid = 10; o_rng = np.linspace(5, 40, grid); s_rng = np.linspace(10, 50, grid); z = np.zeros((grid, grid))
        for i, o in enumerate(o_rng):
            for j, s in enumerate(s_rng):
                z[i,j] = stab_model.predict(inputs)[0]
        st.plotly_chart(px.imshow(z, x=s_rng, y=o_rng, labels=dict(x="Smix %", y="Oil %"), title="System Stability Map"), use_container_width=True)

    if st.button("🔄 Start New Analysis"):
        st.session_state.step = "Step 1: Chemical Setup"
        st.rerun()
