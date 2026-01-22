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
from rdkit.Chem import Draw, Descriptors

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v26.0", layout="wide")

# --- PHARMA-PHYSICS CONSTANTS ---
HLB_DATA = {
    "Tween 80": 15.0, "Tween 20": 16.7, "Span 80": 4.3, "Span 20": 8.6, 
    "Cremophor EL": 13.5, "Labrasol": 14.0, "Solutol HS15": 15.0, "Unknown": 10.0
}

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-top: 4px solid #007bff; text-align: center; margin-bottom: 15px; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE DATA ENGINE ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    # Handle initial state to prevent FileNotFoundError
    if uploaded_file is None:
        return None, None, None, None
        
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None, None, None, None
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: 
        df[f'{col}_clean'] = df[col].apply(get_num)
    
    df['HLB'] = df['Surfactant'].map(HLB_DATA).fillna(10.0)
    
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol}
    except: return None

# --- SESSION STATE INITIALIZATION ---
for key in ['history', 'csv_data', 'drug_name', 'logp', 'mw', 'oil_choice', 's_final', 'cs_final', 'oil_p', 'smix_p']:
    if key not in st.session_state:
        if key == 'history': st.session_state[key] = []
        elif key in ['logp', 'mw']: st.session_state[key] = 3.5
        elif key in ['oil_p', 'smix_p']: st.session_state[key] = 15
        else: st.session_state[key] = "Unknown"

# --- SIDEBAR: TECHNICAL DEFENSE PANE ---
with st.sidebar:
    st.title("🛡️ Conference Hub")
    
    with st.expander("📖 Scientific Encyclopedia"):
        st.markdown("**Decoding for Non-Experts:**")
        st.write("• **PDI**: Uniformity of droplets. <0.2 is ideal.")
        st.write("• **Zeta Potential**: Surface charge. ±30mV prevents clumping.")
        st.write("• **HLB**: Surfactant balance. High HLB = better O/W stability.")
        st.write("• **FaSSIF**: Stability in human gut-simulated fluids.")

    with st.expander("📄 Technical Whitepaper"):
        st.markdown("### Mathematical Foundations")
        st.write("**1. Particle Viscosity**")
        st.latex(r"\eta_{eff} = \eta_0 (1 + 2.5\phi + 6.2\phi^2)")
        
        st.write("**2. Release Kinetics**")
        st.latex(r"M_t / M_\infty = K \cdot t^{1/2}")
        st.caption("Derived from Higuchi Diffusion Theory")
        
        st.write("**3. Thermodynamic Stability**")
        st.latex(r"G_{sys} = \sum \gamma_i A_i - TS_{conf}")

    st.write("---")
    nav_steps = ["Step 1: Setup", "Step 2: AI Selection", "Step 3: Characterization"]
    step_choice = st.radio("SOP Steps", nav_steps)

# --- STEP 1: SETUP ---
if step_choice == "Step 1: Setup":
    st.header("Step 1: Molecular Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Input")
        up = st.file_uploader("Upload Training CSV (Required)", type="csv")
        if up: st.session_state.csv_data = up
        
        smiles = st.text_input("Drug SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        info = get_chem_info(smiles)
        if info:
            st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
            st.image(Draw.MolToImage(info['mol'], size=(300,250)), caption=f"API Structure (LogP: {info['logp']})")
    
    with c2:
        st.subheader("Formulation Ratios")
        st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
        st.session_state.smix_p = st.slider("S-Mix %", 10, 60, 30)

# --- STEP 2: AI SELECTION ---
elif step_choice == "Step 2: AI Selection":
    st.header("Step 2: AI-Driven Screening")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    if le_dict:
        st.session_state.oil_choice = st.selectbox("Select Oil", sorted(le_dict['Oil_phase'].classes_))
        st.session_state.s_final = st.selectbox("Select Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Select Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        
        current_hlb = HLB_DATA.get(st.session_state.s_final, 10.0)
        st.info(f"AI Insight: Using {st.session_state.s_final} (HLB {current_hlb}) for optimized interfacial curvature.")
    else:
        st.error("Engine Offline: Please upload a CSV in Step 1 to train the AI.")

# --- STEP 3: CHARACTERIZATION ---
elif step_choice == "Step 3: Characterization":
    st.header("Step 3: Multi-Parameter Analytics")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    if le_dict:
        def safe_enc(le, val):
            try: return le.transform([val])[0]
            except: return 0

        current_hlb = HLB_DATA.get(st.session_state.s_final, 10.0)
        input_vector = [[safe_enc(le_dict['Drug_Name'], "Unknown"), 
                         safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice), 
                         safe_enc(le_dict['Surfactant'], st.session_state.s_final), 
                         safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final), 
                         current_hlb]]

        preds = {col: models[col].predict(input_vector)[0] for col in models}
        
        # Physics-Based Predictions
        visc = 1.002 * (1 + 2.5 * (st.session_state.oil_p/100))
        t50 = (st.session_state.logp * 4.2) / (1 + (st.session_state.smix_p/100))
        trans = np.clip(100 - (preds['Size_nm'] * 0.14) - (st.session_state.oil_p * 0.3), 0, 100)
        fassif = "High" if (current_hlb > 13 and preds['Size_nm'] < 150) else "Moderate"

        st.subheader("I. Critical Quality Attributes (CQA)")
        c1, c2, c3, c4 = st.columns(4)
        cqas = [("Size", f"{preds['Size_nm']:.1f} nm"), ("PDI", f"{preds['PDI']:.3f}"), 
                ("Zeta", f"{preds['Zeta_mV']:.1f} mV"), ("EE %", f"{preds['Encapsulation_Efficiency']:.1f}%")]
        for i, (l, v) in enumerate(cqas):
            c1, c2, c3, c4 = st.columns(4) if i==0 else (c1, c2, c3, c4) # Placeholder logic
            [c1, c2, c3, c4][i].markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)
        
        st.subheader("II. Advanced Pharmaceutical Performance")
        r1, r2, r3, r4 = st.columns(4)
        adv = [("Viscosity", f"{visc:.2f} cP"), ("t50 Release", f"{t50:.1f} h"), 
               ("Transmittance", f"{trans:.1f}%"), ("FaSSIF", fassif)]
        for i, (l, v) in enumerate(adv):
            [r1, r2, r3, r4][i].markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        st.write("---")
        # Visual Analytics
        col_plot, col_res = st.columns([1.5, 1])
        with col_plot:
            st.subheader("III. Interfacial Phase Space")
            offset = st.session_state.logp * 1.1
            fig = go.Figure(go.Scatterternary({
                'mode': 'lines', 'fill': 'toself', 'name': 'Stability Region',
                'a': [5, 15, 25, 10], 'b': [35+offset, 55+offset, 45+offset, 35+offset], 'c': [60, 30, 30, 55]
            }))
            is_stable = stab_model.predict(input_vector)[0] == 1 and st.session_state.smix_p > st.session_state.oil_p
            fig.add_trace(go.Scatterternary({
                'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 
                'c': [100-st.session_state.oil_p-st.session_state.smix_p],
                'marker': {'size': 14, 'color': 'green' if is_stable else 'red'}
            }))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_res:
            st.subheader("IV. Final Verdict")
            if is_stable:
                st.success("✅ FORMULATION VALIDATED: Thermodynamic spontaneity predicted.")
                            else:
                st.error("⚠️ INSTABILITY RISK: Interfacial film rigidity insufficient. Increase S-Mix ratio.")
                    else:
        st.error("Engine Offline: Please upload a CSV in Step 1.")
