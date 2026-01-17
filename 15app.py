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
st.set_page_config(page_title="NanoPredict AI v19.1 - Stability Fix", layout="wide")

# --- DATABASE & PARAMETERS ---
OIL_HSP = {
    "Capryol 90": [15.8, 8.2, 10.4], "Oleic Acid": [16.4, 3.3, 5.5],
    "Castor Oil": [16.1, 5.2, 9.9], "Olive Oil": [16.5, 3.1, 4.8],
    "Labrafac": [15.7, 5.6, 8.0], "Isopropyl Myristate": [16.2, 3.9, 3.7]
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
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & AI ENGINE (CRASH FIX APPLIED) ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error("Database file missing."); st.stop()
    df = pd.read_csv(csv_file)
    
    # FIX: Handle NaN values in categorical columns immediately
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Specified").astype(str)

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df_raw, models, stab_model, le_dict = load_and_prep()

# --- STRUCTURAL HELPERS ---
def analyze_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    fg = []
    if Fragments.fr_Al_OH(mol) > 0: fg.append("Alcohol (-OH)")
    if Fragments.fr_Ar_OH(mol) > 0: fg.append("Phenol")
    if Fragments.fr_NH2(mol) > 0: fg.append("Primary Amine")
    if Fragments.fr_C_O(mol) > 0: fg.append("Carbonyl")
    if Fragments.fr_COO(mol) > 0: fg.append("Carboxyl/Ester")
    
    # HSP Prediction
    dD = mw / 20.0
    dP = Fragments.fr_Ar_OH(mol) * 4.0 + Fragments.fr_C_O(mol) * 3.0 + 2.0
    dH = Fragments.fr_Al_OH(mol) * 8.0 + Fragments.fr_NH2(mol) * 5.0 + 1.0
    return {"logp": logp, "mw": mw, "fg": fg, "hsp": [dD, dP, dH], "mol": mol}

def find_best_match(smiles):
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return None, 0
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    best_match, max_sim = None, 0
    for drug in df_raw['Drug_Name'].unique():
        if drug == "Not Specified": continue
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
keys = ['drug', 'oil', 'smiles', 'sol_limit', 'oil_p', 'smix_p', 's_final', 'cs_final', 'match_drug', 'logp', 'mw', 'fg']
for k in keys:
    if k not in st.session_state: st.session_state[k] = None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict Pro")
    nav = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    # Fixed sidebar sync
    current_idx = nav.index(st.session_state.step_val) if st.session_state.step_val in nav else 0
    st.session_state.step_val = st.radio("Navigation", nav, index=current_idx)

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API Analysis & Oil Selection")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        mode = st.radio("Drug Input", ["Database API", "New Proprietary SMILES"])
        if mode == "Database API":
            drug_name = st.selectbox("Select Drug", sorted(df_raw['Drug_Name'].unique()))
            smiles = pcp.get_compounds(drug_name, 'name')[0].canonical_smiles
        else:
            smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            drug_name = "New API"

        analysis = analyze_structure(smiles)
        if analysis:
            st.session_state.logp, st.session_state.mw, st.session_state.fg = analysis['logp'], analysis['mw'], analysis['fg']
            st.session_state.smiles, st.session_state.drug = smiles, drug_name
            
            st.write(f"**LogP:** {analysis['logp']:.2f} | **MW:** {analysis['mw']:.2f}")
            st.write(f"**Groups:** {', '.join(analysis['fg']) if analysis['fg'] else 'None'}")
            
            st.write("---")
            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(list(OIL_HSP.keys())))
            if st.button("Confirm Setup →"): go_to_step("Step 2: Concentrations")

    with c2:
        if analysis:
            st.image(Draw.MolToImage(analysis['mol'], size=(400,300)), caption=drug_name)
            drug_hsp = analysis['hsp']
            oil_names, distances = [], []
            for name, hsp in OIL_HSP.items():
                dist = np.sqrt(4*(drug_hsp[0]-hsp[0])**2 + (drug_hsp[1]-hsp[1])**2 + (drug_hsp[2]-hsp[2])**2)
                oil_names.append(name); distances.append(dist)
            
            h_df = pd.DataFrame({"Oil": oil_names, "Dist": distances}).sort_values("Dist")
            st.plotly_chart(px.bar(h_df, x="Dist", y="Oil", orientation='h', title="Solubility Affinity (Heatmap)", color="Dist", color_continuous_scale="RdYlGn_r"), use_container_width=True)
            
            match, score = find_best_match(smiles)
            st.session_state.match_drug = match
            st.success(f"Structural Match: {match} ({score*100:.1f}%)")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Loading & Solubility")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sol_limit = st.number_input(f"Measured Solubility in {st.session_state.oil} (mg/mL)", 0.1, 500.0, 20.0)
        st.session_state.oil_p = st.number_input("Oil %", 5.0, 40.0, 15.0)
        st.session_state.smix_p = st.number_input("S-mix %", 10.0, 60.0, 30.0)
        if st.button("Confirm Ratios →"): go_to_step("Step 3: AI Screening")
    with c2:
        max_load = st.session_state.sol_limit * (st.session_state.oil_p / 100)
        st.metric("Max Drug Loading", f"{max_load:.2f} mg/mL")

# --- STEP 3: SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: AI Component Screening")
    best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    s_list = [s for s in best_data['Surfactant'].unique() if s != "Not Specified"][:5]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rec-box"><b>Recommended Surfactants</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"✅ {s}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rec-box"><b>HLB Properties</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"📊 {s}: {HLB_VALUES.get(s, 10.0)}")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Go to Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION (FIXED SORTING ERROR) ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Formulation Selection")
    c1, c2 = st.columns(2)
    with c1:
        # FIX: Ensure unique values are converted to string and sorted properly
        s_options = sorted([str(s) for s in df_raw['Surfactant'].unique()])
        cs_options = sorted([str(cs) for cs in df_raw['Co-surfactant'].unique()])
        
        st.session_state.s_final = st.selectbox("Select Surfactant", s_options)
        st.session_state.cs_final = st.selectbox("Select Co-Surfactant", cs_options)
        
        if st.button("Generate AI Results →"): go_to_step("Step 5: Results")
    with c2:
        st.markdown(f"""<div class="summary-table"><h4>📋 Formulation Summary</h4>
            API: {st.session_state.drug}<br>
            Oil: {st.session_state.oil} ({st.session_state.oil_p}%)<br>
            Smix: {st.session_state.smix_p}%</div>""", unsafe_allow_html=True)

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: Performance & Kinetic Analysis")
    # AI uses the structural match drug for new APIs
    target = st.session_state.match_drug if st.session_state.match_drug else st.session_state.drug
    
    try:
        inputs = [[le_dict['Drug_Name'].transform([target])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = {col: models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']}
        
        cols = st.columns(4)
        m_data = [("Size", f"{res['Size_nm']:.1f} nm"), ("PDI", f"{res['PDI']:.3f}"), ("Zeta", f"{res['Zeta_mV']:.1f} mV"), ("EE %", f"{res['Encapsulation_Efficiency']:.1f}%")]
        for i, (l, v) in enumerate(m_data):
            with cols[i]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        t1, t2 = st.tabs(["Stability & Phase", "Release Profile"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
                fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'color': '#28a745', 'size': 18, 'symbol': 'diamond'}}))
                fig.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                grid = 10; o_rng = np.linspace(5, 40, grid); s_rng = np.linspace(10, 50, grid); z = np.zeros((grid, grid))
                for i, o in enumerate(o_rng):
                    for j, s in enumerate(s_rng): z[i,j] = stab_model.predict(inputs)[0]
                st.plotly_chart(px.imshow(z, x=s_rng, y=o_rng, title="Stability Zone"), use_container_width=True)
        with t2:
            time = np.linspace(0, 24, 50)
            kh = (12 - (st.session_state.logp or 5)) * (100 / res['Size_nm'])
            rel = np.clip(kh * np.sqrt(time), 0, 100)
            st.plotly_chart(px.line(x=time, y=rel, labels={'x':'Time (h)', 'y':'% Release'}, title="Higuchi Release Profile"), use_container_width=True)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}. Please ensure selected API/Ingredients match database categories.")

    if st.button("🔄 New Analysis"): go_to_step("Step 1: Chemical Setup")
