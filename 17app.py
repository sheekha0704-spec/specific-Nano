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
from rdkit.Chem import Draw, Descriptors, Fragments, AllChem, DataStructs
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v20.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; margin-bottom: 20px; }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    .summary-table { background: #1a202c; color: white; padding: 20px; border-radius: 12px; border-left: 8px solid #f59e0b; }
    .rec-box { background: #f8fbff; border: 2px solid #3b82f6; padding: 15px; border-radius: 12px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & AI ENGINE ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Fallback to local file
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            st.error("Data file not found. Please upload a CSV in the sidebar.")
            st.stop()
    
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
    
    # Stability Logic
    if 'Stability' in df_train.columns:
        df_train['is_stable'] = df_train['Stability'].astype(str).str.lower().str.contains('stable').astype(int)
    else:
        df_train['is_stable'] = 1
    
    stab_model = RandomForestClassifier(n_estimators=100).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- STRUCTURAL HELPERS ---
def analyze_structure(smiles):
    if not smiles: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        logp, mw = Descriptors.MolLogP(mol), Descriptors.MolWt(mol)
        # Hansen Estimate
        dD, dP, dH = mw/20.0, Fragments.fr_Ar_OH(mol)*4.0 + 2.0, Fragments.fr_Al_OH(mol)*8.0 + 1.0
        return {"logp": logp, "mw": mw, "hsp": [dD, dP, dH], "mol": mol}
    except: return None

@st.cache_data
def find_best_match(smiles, drugs_in_db):
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return drugs_in_db[0]
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    best_match, max_sim = drugs_in_db[0], 0
    # Limited to first 50 for speed if DB is huge
    for drug in drugs_in_db[:50]:
        try:
            target_comp = pcp.get_compounds(drug, 'name')[0]
            target_mol = Chem.MolFromSmiles(target_comp.canonical_smiles)
            sim = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(target_mol, 2))
            if sim > max_sim: 
                max_sim = sim
                best_match = drug
        except: continue
    return best_match

# --- INITIALIZE STATE ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
if 'history' not in st.session_state: st.session_state.history = []
# Initialize defaults to prevent AttributeErrors
for key in ['drug', 'oil', 'logp', 'mw', 'match_drug', 's_final', 'cs_final', 'sol_limit', 'drug_conc', 'oil_p', 'smix_p', 'smix_ratio']:
    if key not in st.session_state: st.session_state[key] = None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict Control")
    up_file = st.file_uploader("📂 Option 3: Upload Custom CSV", type=["csv"])
    df_raw, models, stab_model, le_dict = load_and_prep(up_file)
    
    nav = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Navigation", nav, index=nav.index(st.session_state.step_val))
    
    if st.session_state.history:
        st.write("---")
        st.write("📜 **Recent Formulations**")
        for h in st.session_state.history[-5:]: st.info(h)

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API Analysis & Oil Selection")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        mode = st.radio("Drug Input", ["Database API", "New API (SMILES)"])
        if mode == "Database API":
            drug_name = st.selectbox("Select API", sorted(le_dict['Drug_Name'].classes_))
            st.session_state.drug = drug_name
            # Cache SMILES lookup to prevent ServerBusyError
            @st.cache_data
            def get_db_smiles(name):
                try: return pcp.get_compounds(name, 'name')[0].canonical_smiles
                except: return "CC(=O)OC1=CC=CC=C1C(=O)O"
            smiles = get_db_smiles(drug_name)
        else:
            smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            st.session_state.drug = "New API"

        analysis = analyze_structure(smiles)
        if analysis:
            st.session_state.logp = analysis['logp']
            st.session_state.mw = analysis['mw']
            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
            st.session_state.match_drug = find_best_match(smiles, le_dict['Drug_Name'].classes_)
            
            if st.button("Confirm Setup →"): go_to_step("Step 2: Concentrations")
    with c2:
        if analysis:
            st.image(Draw.MolToImage(analysis['mol'], size=(400,250)), caption="Molecular Structure")
            hsp = analysis['hsp']
            st.write(f"**Solubility Profile ($\delta$):** D:{hsp[0]:.1f} P:{hsp[1]:.1f} H:{hsp[2]:.1f}")
            
# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Loading & Solubility Prediction")
    # AI Equation
    pred_sol = 10**(0.5 - (0.01 * (st.session_state.mw - 50)) - (0.6 * st.session_state.logp)) * 1000
    st.session_state.sol_limit = np.clip(pred_sol, 1.0, 500.0)
    
    st.success(f"AI Predicted Solubility in {st.session_state.oil}: **{st.session_state.sol_limit:.2f} mg/mL**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.drug_conc = st.number_input("Target Drug Conc (mg/mL)", 0.1, 500.0, 2.0)
        st.session_state.oil_p = st.slider("Oil %", 5.0, 40.0, 15.0)
    with c2:
        st.session_state.smix_p = st.slider("S-mix %", 10.0, 60.0, 30.0)
        st.session_state.smix_ratio = st.selectbox("S-mix Ratio (S:Co-S)", ["1:1", "2:1", "3:1", "1:2"])

    if st.button("Confirm Ratios →"): go_to_step("Step 3: AI Screening")

# --- STEP 3: SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: AI Component Screening")
    best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    s_list = [s for s in best_data['Surfactant'].unique() if s != "Not Specified"][:5]
    
    st.markdown('<div class="rec-box"><b>AI Ranked Surfactants</b>', unsafe_allow_html=True)
    for s in s_list: st.write(f"✅ {s}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Go to Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Formulation Selection")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.s_final = st.selectbox("Final Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Final Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        if st.button("Generate AI Results →"): go_to_step("Step 5: Results")
    with c2:
        st.markdown(f"""<div class="summary-table"><h4>📋 Parameters</h4>
            API: {st.session_state.drug} (MW: {st.session_state.mw:.1f})<br>
            Oil: {st.session_state.oil} ({st.session_state.oil_p}%)<br>
            S-mix: {st.session_state.smix_p}% ({st.session_state.smix_ratio})<br>
            Drug Loading: {st.session_state.drug_conc} mg/mL</div>""", unsafe_allow_html=True)

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: AI Performance Suite")
    
    # Use match_drug to prevent Encoding Error for "New API"
    pred_api = st.session_state.match_drug if st.session_state.drug == "New API" else st.session_state.drug
    
    try:
        inputs = [[le_dict['Drug_Name'].transform([pred_api])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = {col: models[col].predict(inputs)[0] for col in models}
        stable_val = stab_model.predict(inputs)[0]

        # SIX OUTPUTS
        m_data = [
            ("Size", f"{res['Size_nm']:.1f} nm"), 
            ("PDI", f"{res['PDI']:.3f}"), 
            ("Zeta", f"{res['Zeta_mV']:.1f} mV"),
            ("EE %", f"{res['Encapsulation_Efficiency']:.1f}%"), 
            ("Solubility", f"{st.session_state.sol_limit:.1f}"),
            ("Stability", "Stable" if stable_val == 1 else "Unstable")
        ]
        
        cols = st.columns(3)
        for i, (l, v) in enumerate(m_data):
            with cols[i % 3]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        # PLOTS
        st.write("### Phase Analysis")
        water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers', 
            'a': [st.session_state.oil_p], 
            'b': [st.session_state.smix_p], 
            'c': [water_p], 
            'marker': {'size': 20, 'color': 'red', 'symbol': 'diamond'}
        }))
        fig.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}})
        st.plotly_chart(fig, use_container_width=True)
        
        # SIZE CURVE
        mu, sigma = res['Size_nm'], res['Size_nm'] * (res['PDI'] + 0.05)
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y = np.exp(-0.5 * ((x - mu)/sigma)**2)
        st.plotly_chart(px.line(x=x, y=y, title="Predicted Particle Size Distribution"), use_container_width=True)
        
        if f"{st.session_state.drug} ({st.session_state.oil})" not in st.session_state.history:
            st.session_state.history.append(f"{st.session_state.drug} ({st.session_state.oil})")

    except Exception as e:
        st.error(f"AI Prediction Error: {e}")

    if st.button("🔄 New Formulation"): go_to_step("Step 1: Chemical Setup")
