import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v27.0", layout="wide")

# --- PHARMA-PHYSICS DATABASE ---
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
    .advice-box { background: #f0f7ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE (FIXED LOAD LOGIC) ---
@st.cache_resource
def load_and_prep(uploaded_file):
    if uploaded_file is None:
        return None, None, None, None
    
    try:
        df = pd.read_csv(uploaded_file)
        cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
        
        # Data Cleaning
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

        def get_num(x):
            val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
            return float(val[0]) if val else 0.0

        targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
        for col in targets:
            df[f'{col}_clean'] = df[col].apply(get_num)
        
        df['HLB'] = df['Surfactant'].map(HLB_DATA).fillna(10.0)
        
        # Encoding
        le_dict = {}
        df_train = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
            le_dict[col] = le

        # Training
        X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']]
        models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
        
        df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
        stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
        
        return df, models, stab_model, le_dict
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return None, None, None, None

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol}
    except: return None

# --- SESSION STATE ---
for key in ['history', 'csv_data', 'logp', 'mw', 'oil_p', 'smix_p']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'history' else (3.5 if key == 'logp' else 15)

# --- SIDEBAR: KNOWLEDGE HUB & DECODER ---
with st.sidebar:
    st.title("🔬 Conference Control")
    
    with st.expander("📖 Scientific Decoder (Glossary)"):
        st.write("**PDI**: Measures droplet size uniformity. <0.2 is ideal.")
        st.write("**Zeta Potential**: Charge stability. >±30mV is stable.")
        st.write("**HLB**: Balance of surfactant. High = Oil-in-Water.")
        st.write("**FaSSIF**: Human gut simulation for stability.")

    with st.expander("📄 Technical Whitepaper (Mathematics)"):
        st.write("**1. Effective Viscosity**")
        st.latex(r"\eta = \eta_0 (1 + 2.5\phi)")
        st.write("**2. Release Rate (Higuchi)**")
        st.latex(r"Q = K \sqrt{t}")
        st.write("**3. Stability (Gibbs)**")
        st.latex(r"\Delta G = \gamma \Delta A - T \Delta S")

    st.write("---")
    nav = st.radio("Navigation", ["1. Data & Molecular Setup", "2. AI Prediction Suite"])

# --- MAIN INTERFACE ---
if nav == "1. Data & Molecular Setup":
    st.header("Step 1: System Initialization")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Training Data")
        up = st.file_uploader("Upload CSV", type="csv")
        if up: st.session_state.csv_data = up
        
        smiles = st.text_input("API SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        info = get_chem_info(smiles)
        if info:
            st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
            st.image(Draw.MolToImage(info['mol'], size=(300,250)), caption=f"LogP: {info['logp']}")

    with c2:
        st.subheader("Physical Ratios")
        st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
        st.session_state.smix_p = st.slider("S-Mix %", 10, 60, 30)

else:
    st.header("Step 2: AI Multi-Parameter Analytics")
    df, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    if df is not None:
        # Selection UI
        c_sel1, c_sel2, c_sel3 = st.columns(3)
        oil = c_sel1.selectbox("Oil Phase", sorted(le_dict['Oil_phase'].classes_))
        surf = c_sel2.selectbox("Surfactant", sorted(le_dict['Surfactant'].classes_))
        cosurf = c_sel3.selectbox("Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))

        # Prediction Execution
        def encode_val(le, val):
            return le.transform([val])[0] if val in le.classes_ else 0

        h_val = HLB_DATA.get(surf, 10.0)
        X_input = [[encode_val(le_dict['Drug_Name'], "Unknown"), encode_val(le_dict['Oil_phase'], oil), 
                    encode_val(le_dict['Surfactant'], surf), encode_val(le_dict['Co-surfactant'], cosurf), h_val]]
        
        preds = {col: models[col].predict(X_input)[0] for col in models}
        
        # Physics Engines
        visc = 1.002 * (1 + 2.5 * (st.session_state.oil_p/100))
        t50 = (st.session_state.logp * 4.2) / (1 + (st.session_state.smix_p/100))
        trans = np.clip(100 - (preds['Size_nm'] * 0.15), 0, 100)
        fassif = "High" if (h_val > 12 and preds['Size_nm'] < 200) else "Moderate"

        # RESULTS GRID (All 8 Parameters)
        st.subheader("Comprehensive Characterization")
        r1 = st.columns(4)
        cqa_data = [("Size", f"{preds['Size_nm']:.1f} nm"), ("PDI", f"{preds['PDI']:.3f}"), 
                    ("Zeta", f"{preds['Zeta_mV']:.1f} mV"), ("EE%", f"{preds['Encapsulation_Efficiency']:.1f}%")]
        for i, (l, v) in enumerate(cqa_data):
            r1[i].markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)
            
        r2 = st.columns(4)
        perf_data = [("Viscosity", f"{visc:.2f} cP"), ("t50 Release", f"{t50:.1f} h"), 
                     ("Transmittance", f"{trans:.1f}%"), ("FaSSIF Stability", fassif)]
        for i, (l, v) in enumerate(perf_data):
            r2[i].markdown(f"<div class='metric-card' style='border-top-color:#28a745;'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        # PLOTS
        st.write("---")
        cp1, cp2 = st.columns([1.5, 1])
        with cp1:
            st.subheader("Interfacial Phase Space")
            offset = st.session_state.logp * 1.1
            fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stability Region',
                'a': [5, 15, 25, 10], 'b': [35+offset, 55+offset, 45+offset, 35+offset], 'c': [60, 30, 30, 55]}))
            is_stable = stab_model.predict(X_input)[0] == 1 and st.session_state.smix_p > st.session_state.oil_p
            fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 
                'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 14, 'color': 'green' if is_stable else 'red'}}))
            st.plotly_chart(fig, use_container_width=True)
        
        with cp2:
            st.subheader("AI Verdict")
            if is_stable: st.success("✅ OPTIMIZED: Thermodynamic spontaneity predicted.")
            else: st.error("⚠️ RISK: Interfacial film rigidity insufficient.")
            st.info("💡 Use the Sidebar Decoder to explain these results to your audience.")
    else:
        st.warning("Please upload a CSV file in Step 1 to begin.")
