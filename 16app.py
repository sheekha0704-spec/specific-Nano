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
st.set_page_config(page_title="NanoPredict Pro v25.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; }
    .m-label { font-size: 12px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; color: #1a202c; font-weight: 800; }
    .summary-table { background: #1a202c; color: white; padding: 15px; border-radius: 12px; border-left: 8px solid #f59e0b; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE (HANDLES CUSTOM UPLOADS) ---
@st.cache_resource
def load_and_prep(file):
    # Determine which file to use
    if file is not None:
        df = pd.read_csv(file)
    else:
        # Standard default file
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            st.error("Please upload a CSV file in the sidebar to begin."); st.stop()
    
    # Clean Categorical Data
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Not Specified").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    # Target features for prediction
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    # Label Encoders
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    # AI Training
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    # Stability Model
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).astype(str).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

# --- SIDEBAR: CSV UPLOAD & HISTORY ---
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
    
    # Re-load data if a new file is uploaded
    df_raw, models, stab_model, le_dict = load_and_prep(uploaded_file)
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if st.session_state.history:
        st.header("📜 Formulation History")
        for entry in reversed(st.session_state.history[-5:]):
            st.info(f"Drug: {entry['drug']}\nOil: {entry['oil']}")

# --- STRUCTURAL MATCHING ENGINE ---
def find_best_match(smiles):
    query_mol = Chem.MolFromSmiles(smiles)
    if not query_mol: return None, 0
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2)
    best_match, max_sim = None, 0
    # Search database for similar moiety
    for drug in le_dict['Drug_Name'].classes_:
        if drug in ["Not Specified", "New API"]: continue
        try:
            # Note: In a production environment, pre-caching these fingerprints is recommended
            comp = pcp.get_compounds(drug, 'name')[0]
            target_mol = Chem.MolFromSmiles(comp.canonical_smiles)
            sim = DataStructs.TanimotoSimilarity(query_fp, AllChem.GetMorganFingerprintAsBitVect(target_mol, 2))
            if sim > max_sim: max_sim, best_match = sim, drug
        except: continue
    return best_match, max_sim

# --- NAVIGATION ---
if 'step' not in st.session_state: st.session_state.step = 1

def set_step(s):
    st.session_state.step = s
    st.rerun()

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step == 1:
    st.header("Step 1: Unknown Compound Setup")
    c1, c2 = st.columns([1, 1])
    with c1:
        smiles = st.text_input("Enter Compound SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.session_state.smiles = smiles
            st.session_state.logp = Descriptors.MolLogP(mol)
            st.session_state.mw = Descriptors.MolWt(mol)
            st.session_state.match_drug, score = find_best_match(smiles)
            
            st.success(f"Analyzed: LogP {st.session_state.logp:.2f} | MW {st.session_state.mw:.1f}")
            st.image(Draw.MolToImage(mol, size=(300, 200)), caption="Structural moiety analyzed")
            
            st.session_state.oil = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
            if st.button("Proceed to Solubility Analysis →"): set_step(2)
        else:
            st.error("Invalid SMILES. Please check your chemical string.")

# --- STEP 2: AUTO-SOLUBILITY ---
elif st.session_state.step == 2:
    st.header("Step 2: Predicted Solubility & Loading")
    # AI-based solubility estimation based on LogP
    est_sol = np.clip(450 / (st.session_state.logp + 0.5), 2.0, 500.0)
    
    st.session_state.sol_limit = st.slider("Predicted Solubility (mg/mL)", 1.0, 500.0, float(est_sol))
    st.session_state.drug_conc = st.number_input("Target Drug Concentration (mg/mL)", 0.1, st.session_state.sol_limit, 2.0)
    
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    if st.button("Confirm Ratios →"): set_step(3)

# --- STEP 3: SCREENING ---
elif st.session_state.step == 3:
    st.header("Step 3: AI Ingredient Screening")
    relevant = df_raw[df_raw['Oil_phase'] == st.session_state.oil]
    top_s = relevant.sort_values('Encapsulation_Efficiency_clean', ascending=False).head(5)
    
    st.write("Top surfactants for your selection based on database EE%:")
    st.dataframe(top_s[['Surfactant', 'Encapsulation_Efficiency']])
    
    if st.button("Proceed to Final Selection →"): set_step(4)

# --- STEP 4: SELECTION ---
elif st.session_state.step == 4:
    st.header("Step 4: Final Ingredients")
    # Fix: Ensure all options are strings to prevent sorting TypeErrors
    s_options = sorted([str(s) for s in le_dict['Surfactant'].classes_])
    cs_options = sorted([str(cs) for cs in le_dict['Co-surfactant'].classes_])
    
    st.session_state.s_final = st.selectbox("Final Surfactant", s_options)
    st.session_state.cs_final = st.selectbox("Final Co-Surfactant", cs_options)
    
    if st.button("Run Global AI Predictor →"): set_step(5)

# --- STEP 5: RESULTS (6 MAJOR OUTPUTS + PLOTS) ---
elif st.session_state.step == 5:
    st.header("Step 5: AI Performance Suite")
    
    # Bridge: Use structural similarity match to prevent "New API" label error
    proxy_drug = st.session_state.match_drug if st.session_state.match_drug else le_dict['Drug_Name'].classes_[0]
    
    try:
        inputs = [[le_dict['Drug_Name'].transform([proxy_drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        preds = {k: models[k].predict(inputs)[0] for k in models}
        is_stable = stab_model.predict(inputs)[0]

        # 1. SIX MAJOR OUTPUTS
        m_cols = st.columns(3)
        m_cols[0].markdown(f'<div class="metric-card"><div class="m-label">Size</div><div class="m-value">{preds["Size_nm"]:.1f} nm</div></div>', unsafe_allow_html=True)
        m_cols[1].markdown(f'<div class="metric-card"><div class="m-label">PDI</div><div class="m-value">{preds["PDI"]:.3f}</div></div>', unsafe_allow_html=True)
        m_cols[2].markdown(f'<div class="metric-card"><div class="m-label">Zeta</div><div class="m-value">{preds["Zeta_mV"]:.1f} mV</div></div>', unsafe_allow_html=True)
        
        m_cols2 = st.columns(3)
        m_cols2[0].markdown(f'<div class="metric-card"><div class="m-label">EE%</div><div class="m-value">{preds["Encapsulation_Efficiency"]:.1f}%</div></div>', unsafe_allow_html=True)
        m_cols2[1].markdown(f'<div class="metric-card"><div class="m-label">Solubility</div><div class="m-value">{st.session_state.sol_limit:.1f} mg/mL</div></div>', unsafe_allow_html=True)
        status_text = "Stable" if is_stable == 1 else "Unstable"
        m_cols2[2].markdown(f'<div class="metric-card"><div class="m-label">Stability</div><div class="m-value">{status_text}</div></div>', unsafe_allow_html=True)

        # 2. VISUALIZATIONS
        t1, t2 = st.tabs(["Particle Analysis", "Ternary & Loading"])
        with t1:
            # Distribution Curve (Gaussian Approximation)
            mu, sigma = preds['Size_nm'], preds['Size_nm'] * preds['PDI']
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu)/sigma)**2)
            fig_dist = px.line(x=x, y=y, title="Predicted Particle Size Distribution Curve", labels={'x':'Size (nm)', 'y':'Probability Density'})
            st.plotly_chart(fig_dist, use_container_width=True)
            

[Image of particle size distribution curve]


        with t2:
            c_a, c_b = st.columns(2)
            # Ternary Diagram
            fig_tern = go.Figure(go.Scatterternary({
                'mode': 'markers',
                'a': [st.session_state.oil_p],
                'b': [st.session_state.smix_p],
                'c': [100 - st.session_state.oil_p - st.session_state.smix_p],
                'marker': {'color': 'blue', 'size': 15, 'symbol': 'circle'}
            }))
            fig_tern.update_layout(title="Ternary Phase Distribution", ternary={'sum': 100, 'aaxis':{'title':'Oil%'}, 'baxis':{'title':'Smix%'}})
            c_a.plotly_chart(fig_tern, use_container_width=True)
            
            
            # Drug loading bar
            c_b.write("### Drug Incorporation")
            c_b.progress(st.session_state.drug_conc / st.session_state.sol_limit)
            c_b.write(f"Used {st.session_state.drug_conc} mg/mL out of {st.session_state.sol_limit:.1f} mg/mL capacity")

        # Save to history
        if not any(h['drug'] == proxy_drug for h in st.session_state.history):
            st.session_state.history.append({"drug": proxy_drug, "oil": st.session_state.oil})

    except Exception as e:
        st.error(f"Prediction logic error: {e}")

    if st.button("🔄 Start New Formulation"): set_step(1)
