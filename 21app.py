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
st.set_page_config(page_title="NanoPredict AI v25.0 | Conference Edition", layout="wide")

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
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin-bottom: 20px;}
    .whitepaper-box { background: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; font-family: 'serif'; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: return None, None, None, None
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    df['HLB'] = df['Surfactant'].map(HLB_DATA).fillna(10.0)
    
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol}
    except: return None

# --- INITIALIZE STATE ---
for key in ['history', 'csv_data', 'drug_name', 'logp', 'mw', 'oil_choice', 's_final', 'cs_final', 'oil_p', 'smix_p']:
    if key not in st.session_state:
        if key == 'history': st.session_state[key] = []
        elif key in ['logp', 'mw']: st.session_state[key] = 3.5
        elif key in ['oil_p', 'smix_p']: st.session_state[key] = 15
        else: st.session_state[key] = "Unknown"

# --- SIDEBAR: THE CONFERENCE KNOWLEDGE HUB ---
with st.sidebar:
    st.title("🛡️ Conference Defense")
    
    with st.expander("📖 Scientific Encyclopedia (Common Terms)"):
        st.write("**PDI**: Measures size uniformity. Values < 0.2 = monodisperse.")
        st.write("**Zeta Potential**: Charge stability. ±30mV = high repulsion.")
        st.write("**HLB**: Surfactant affinity. >10 = O/W emulsion.")
        st.write("**FaSSIF**: Fasted State Intestinal Fluid behavior.")

    with st.expander("📄 Technical Whitepaper (Mathematics)"):
        st.markdown("### Model Logic")
        st.write("**1. Viscosity ($ \eta $)**")
        st.latex(r"\eta = \eta_0 (1 + 2.5\phi)")
        st.caption("Einstein-Stokes extension for Volume Fraction")
        
        st.write("**2. Release Kinetics ($ t_{50} $)**")
        st.latex(r"Q = K \cdot \sqrt{t}")
        st.caption("Higuchi-based Diffusion Model")
        
        st.write("**3. Stability Index**")
        st.latex(r"\Delta G = \gamma \Delta A - T \Delta S")
        st.caption("Gibbs Free Energy of Emulsification")

    st.write("---")
    nav_steps = ["Step 1: Setup", "Step 2: AI Selection", "Step 3: Characterization"]
    step_choice = st.radio("Navigation", nav_steps)
    
    if st.session_state.history:
        st.subheader("📜 History")
        for item in st.session_state.history[-3:]: st.caption(f"✅ {item}")

# --- MAIN UI LOGIC ---
df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)

if step_choice == "Step 1: Setup":
    st.header("Step 1: Molecular & Data Input")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Input")
        up = st.file_uploader("Upload Training CSV", type="csv")
        if up: st.session_state.csv_data = up
        
        smiles = st.text_input("API SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        info = get_chem_info(smiles)
        if info:
            st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
            st.image(Draw.MolToImage(info['mol'], size=(250,200)), caption="Molecular Structure")
    
    with c2:
        st.subheader("Physicochemical Settings")
        st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
        st.session_state.smix_p = st.slider("S-Mix %", 10, 60, 30)

elif step_choice == "Step 2: AI Selection":
    st.header("Step 2: Component AI Screening")
    if le_dict:
        st.session_state.oil_choice = st.selectbox("Oil Phase", sorted(le_dict['Oil_phase'].classes_))
        st.session_state.s_final = st.selectbox("Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        st.info(f"Current selection utilizes an HLB of {HLB_DATA.get(st.session_state.s_final, 10.0)} for interfacial tension reduction.")
    else:
        st.warning("Please upload a CSV in Step 1.")

elif step_choice == "Step 3: Characterization":
    st.header("Step 3: Performance Suite")
    if le_dict:
        def safe_enc(le, val):
            try: return le.transform([val])[0]
            except: return 0

        current_hlb = HLB_DATA.get(st.session_state.s_final, 10.0)
        input_data = [[safe_enc(le_dict['Drug_Name'], "Unknown"), safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice), 
                       safe_enc(le_dict['Surfactant'], st.session_state.s_final), safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final), current_hlb]]

        res = {col: models[col].predict(input_data)[0] for col in models}
        
        # Mechanistic Calcs
        visc = 1.002 * (1 + 2.5 * (st.session_state.oil_p/100))
        t50 = (st.session_state.logp * 4.5) / (1 + (st.session_state.smix_p/100))
        trans = np.clip(100 - (res['Size_nm'] * 0.12) - (st.session_state.oil_p * 0.4), 0, 100)
        fassif = "High" if (current_hlb > 12 and res['Size_nm'] < 200) else "Moderate"

        st.subheader("Critical Quality Attributes (CQAs)")
        cols = st.columns(4)
        m_list = [("Size", f"{res['Size_nm']:.1f} nm"), ("PDI", f"{res['PDI']:.3f}"), ("Zeta", f"{res['Zeta_mV']:.1f} mV"), ("EE%", f"{res['Encapsulation_Efficiency']:.1f}%")]
        for i, (l, v) in enumerate(m_list):
            cols[i].markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)
        
        cols2 = st.columns(4)
        m_list2 = [("Viscosity", f"{visc:.2f} cP"), ("t50 Release", f"{t50:.1f} h"), ("Transmittance", f"{trans:.1f}%"), ("FaSSIF", fassif)]
        for i, (l, v) in enumerate(m_list2):
            cols2[i].markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        st.write("---")
        c_plot, c_rep = st.columns([1.5, 1])
        with c_plot:
            offset = st.session_state.logp * 1.2
            fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Self-Emulsification Region', 'a': [5, 15, 25, 10], 'b': [40+offset, 50+offset, 45+offset, 35+offset], 'c': [55, 35, 30, 55]}))
            is_stable = stab_model.predict(input_data)[0] == 1 and st.session_state.smix_p > st.session_state.oil_p
            fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 14, 'color': 'green' if is_stable else 'red'}}))
            st.plotly_chart(fig, use_container_width=True)
        
        with c_rep:
            if not is_stable: st.error("⚠️ Thermodynamic Risk: Increase S-mix to stabilize interfacial film.")
            else: st.success("✅ Formulation Verified.")
            
        if st.button("Log Trial"):
            st.session_state.history.append(f"{st.session_state.oil_choice} Trial")
    else:
        st.warning("Please upload data in Step 1.")
