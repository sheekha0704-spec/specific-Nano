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
st.set_page_config(page_title="NanoPredict AI v18.0 - Expert Solubility Suite", layout="wide")

# --- DATABASE & PARAMETERS ---
OIL_HSP = {
    "Capryol 90": [15.8, 8.2, 10.4],
    "Oleic Acid": [16.4, 3.3, 5.5],
    "Castor Oil": [16.1, 5.2, 9.9],
    "Olive Oil": [16.5, 3.1, 4.8],
    "Labrafac": [15.7, 5.6, 8.0],
    "Isopropyl Myristate": [16.2, 3.9, 3.7]
}

HLB_VALUES = {
    "Tween 80": 15.0, "Tween 20": 16.7, "Span 80": 4.3, "Span 20": 8.6,
    "Cremophor EL": 13.5, "Solutol HS15": 15.0, "Lecithin": 4.0, "Labrasol": 12.0,
    "Transcutol P": 4.0, "PEG 400": 11.0, "Capmul MCM": 5.5, "Not Specified": 10.0
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; margin-bottom: 20px; }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    .rec-box { background: #f8fbff; border: 2px solid #3b82f6; padding: 15px; border-radius: 12px; margin-bottom: 10px; }
    .summary-table { background: #1a202c; color: white; padding: 20px; border-radius: 12px; border-left: 8px solid #f59e0b; }
    .algo-box { background: #f0f7ff; border: 1px dashed #007bff; padding: 15px; border-radius: 10px; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & AI ENGINE ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error("Database file missing."); st.stop()
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
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df_raw, models, stab_model, le_dict = load_and_prep()

# --- HELPER FUNCTIONS ---
def get_hsp_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return [0, 0, 0]
    dD = Descriptors.MolWt(mol) / 20.0
    dP = Fragments.fr_Ar_OH(mol) * 4.0 + Fragments.fr_C_O(mol) * 3.0 + 2.0
    dH = Fragments.fr_Al_OH(mol) * 8.0 + Fragments.fr_NH2(mol) * 5.0 + 1.0
    return [dD, dP, dH]

def find_best_match(smiles):
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
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
keys = ['drug', 'oil', 'smiles', 'sol_limit', 'oil_p', 'smix_p', 's_final', 'cs_final', 'match_drug', 'logp', 'mw']
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict Pro")
    nav = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Navigation", nav, index=nav.index(st.session_state.step_val))

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API Affinity & Solubility Heatmap")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        mode = st.radio("Input Method", ["Database Drug", "New Proprietary Drug (SMILES)"])
        if mode == "Database Drug":
            drug_name = st.selectbox("Select API", sorted(df_raw['Drug_Name'].unique()))
            comp = pcp.get_compounds(drug_name, 'name')[0]
            smiles = comp.canonical_smiles
        else:
            smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
            drug_name = "New Experimental API"
        
        if st.button("Analyze Structure →"):
            st.session_state.drug, st.session_state.smiles = drug_name, smiles
            st.rerun()

    with c2:
        if st.session_state.smiles:
            mol = Chem.MolFromSmiles(st.session_state.smiles)
            if mol:
                st.image(Draw.MolToImage(mol, size=(400,300)), caption=st.session_state.drug)
                drug_hsp = get_hsp_from_smiles(st.session_state.smiles)
                oil_names, distances = [], []
                for name, hsp in OIL_HSP.items():
                    dist = np.sqrt(4*(drug_hsp[0]-hsp[0])**2 + (drug_hsp[1]-hsp[1])**2 + (drug_hsp[2]-hsp[2])**2)
                    oil_names.append(name); distances.append(dist)
                
                h_df = pd.DataFrame({"Oil Phase": oil_names, "HSP Distance": distances}).sort_values("HSP Distance")
                fig = px.bar(h_df, x="HSP Distance", y="Oil Phase", orientation='h', title="Solubility Affinity (Lower is Better)", color="HSP Distance", color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig, use_container_width=True)
                
                match, score = find_best_match(st.session_state.smiles)
                st.session_state.match_drug = match
                st.session_state.logp = Descriptors.MolLogP(mol)
                st.session_state.mw = Descriptors.MolWt(mol)
                st.info(f"Structural Similarity Match: **{match}** ({score*100:.1f}%)")
                
                st.session_state.oil = h_df.iloc[0]["Oil Phase"]
                if st.button("Confirm Oil Selection & Proceed →"): go_to_step("Step 2: Concentrations")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Solubility Limits & Ratios")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sol_limit = st.number_input(f"Measured Solubility in {st.session_state.oil} (mg/mL)", 0.1, 500.0, 25.0)
        st.session_state.oil_p = st.number_input("Oil %", 5.0, 40.0, 15.0)
        st.session_state.smix_p = st.number_input("S-mix %", 10.0, 60.0, 30.0)
        if st.button("Save & Screen Components →"): go_to_step("Step 3: AI Screening")
    with c2:
        max_load = st.session_state.sol_limit * (st.session_state.oil_p / 100)
        st.metric("Theoretical Max Loading", f"{max_load:.2f} mg/mL")
        st.markdown("<div class='algo-box'><b>Note:</b> Exceeding this limit causes Ostwald Ripening and Drug Precipitation.</div>", unsafe_allow_html=True)

# --- STEP 3: AI SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Component AI Screening")
    best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    s_list = best_data['Surfactant'].unique()[:5]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rec-box"><b>Top Performers</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"✅ {s}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rec-box"><b>HLB Reference</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"📊 {s}: {HLB_VALUES.get(s, 10.0)}")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Go to Final Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Selection & Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.s_final = st.selectbox("Select Surfactant", sorted(df_raw['Surfactant'].unique()))
        st.session_state.cs_final = st.selectbox("Select Co-Surfactant", sorted(df_raw['Co-surfactant'].unique()))
        if st.button("Execute AI Predictor →"): go_to_step("Step 5: Results")
    with c2:
        st.markdown(f"""<div class="summary-table"><h4>📋 Formulation Summary</h4>
            Oil: {st.session_state.oil} ({st.session_state.oil_p}%)<br>
            Smix: {st.session_state.smix_p}%<br>
            API: {st.session_state.drug}</div>""", unsafe_allow_html=True)

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: AI Performance & Kinetic Analysis")
    target_drug = st.session_state.match_drug if st.session_state.match_drug else st.session_state.drug
    
    inputs = [[le_dict['Drug_Name'].transform([target_drug])[0],
               le_dict['Oil_phase'].transform([st.session_state.oil])[0],
               le_dict['Surfactant'].transform([st.session_state.s_final])[0],
               le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
    
    res = {col: models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
    
    cols = st.columns(4)
    m_data = [("Size", f"{res['Size_nm']:.1f} nm"), ("PDI", f"{res['PDI']:.3f}"), ("Zeta", f"{res['Zeta_mV']:.1f} mV"), ("EE %", f"{res['Encapsulation_Efficiency']:.1f}%")]
    for i, (l, v) in enumerate(m_data):
        with cols[i]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Ternary Phase", "Release Profile", "Stability Map"])
    with t1:
        water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'color': '#28a745', 'size': 18, 'symbol': 'diamond'}}))
        fig.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}})
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        time = np.linspace(0, 24, 50)
        kh = (12 - (st.session_state.logp or 5)) * (100 / res['Size_nm'])
        rel = np.clip(kh * np.sqrt(time), 0, 100)
        st.plotly_chart(px.line(x=time, y=rel, title="Simulated Higuchi Release Profile"), use_container_width=True)
    with t3:
        grid = 10; o_rng = np.linspace(5, 40, grid); s_rng = np.linspace(10, 50, grid); z = np.zeros((grid, grid))
        for i, o in enumerate(o_rng):
            for j, s in enumerate(s_rng): z[i,j] = stab_model.predict(inputs)[0]
        st.plotly_chart(px.imshow(z, x=s_rng, y=o_rng, title="Stability Safe-Zone"), use_container_width=True)

    if st.button("🔄 Reset Formulation"): go_to_step("Step 1: Chemical Setup")
