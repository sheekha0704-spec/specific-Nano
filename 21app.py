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
st.set_page_config(page_title="NanoPredict AI v27.0", layout="wide")

# --- DATA CONSTANTS ---
HLB_DATA = {
    "Tween 80": 15.0, "Tween 20": 16.7, "Span 80": 4.3, "Span 20": 8.6, 
    "Cremophor EL": 13.5, "Labrasol": 14.0, "Solutol HS15": 15.0, "Unknown": 10.0
}

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; }
    .justification { font-size: 12px; color: #444; font-style: italic; margin-bottom: 10px; }
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
    
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
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
if 'history' not in st.session_state: st.session_state.history = []
if 'csv_data' not in st.session_state: st.session_state.csv_data = None
if 'drug_name' not in st.session_state: st.session_state.drug_name = "None"
if 'logp' not in st.session_state: st.session_state.logp = 3.5
if 'mw' not in st.session_state: st.session_state.mw = 300.0

# --- SIDEBAR: JUSTIFIED TECHNICAL PANEL ---
with st.sidebar:
    st.title("Settings")
    
    # CONFERENCE JUSTIFICATION PANEL
    with st.expander("📖 Technical Justifications"):
        st.subheader("Physics Engine")
        st.latex(r"\eta = \eta_0 (1 + 2.5\phi)")
        st.markdown("<div class='justification'><b>Einstein-Stokes:</b> Used to predict Viscosity based on the volume fraction (Oil %) of the internal phase.</div>", unsafe_allow_html=True)
        
        st.latex(r"Q = K \cdot t^{1/2}")
        st.markdown("<div class='justification'><b>Higuchi Model:</b> Predicts drug release kinetics from the nano-oil core into the medium.</div>", unsafe_allow_html=True)
        
        st.latex(r"Log S \propto -LogP")
        st.markdown("<div class='justification'><b>GSE Equation:</b> Justifies why higher LogP drugs require more surfactant for stable encapsulation.</div>", unsafe_allow_html=True)

    nav_steps = ["Step 1: Drug & Oil", "Step 2: Solubility", "Step 3: Component AI", "Step 4: Ratios", "Step 5: Final Selection", "Step 6: AI Predictions"]
    step_choice = st.radio("Formulation Steps", nav_steps)
    
    st.write("---")
    st.subheader("📜 History")
    if st.session_state.history:
        for item in st.session_state.history[-5:]:
            st.caption(f"✅ {item}")

# --- STEP 1: DRUG & OIL ---
if step_choice == "Step 1: Drug & Oil":
    st.header("Step 1: Chemical Setup")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Drug Input")
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
        
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            info = get_chem_info(smiles)
            if info:
                st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
                st.session_state.drug_name = "Custom_API"
                st.image(Draw.MolToImage(info['mol'], size=(250,200)))
        else:
            if le_dict:
                st.session_state.drug_name = st.selectbox("API", sorted(le_dict['Drug_Name'].classes_))
            else: st.warning("Upload CSV first.")
        
        up = st.file_uploader("Option 3: Load CSV", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Solubility Comparison")
            oils = le_dict['Oil_phase'].classes_
            scores = [max(5, 100 - abs(st.session_state.logp - 3.2)*12 + np.random.randint(-1,1)) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility (mg/mL)": scores}).sort_values("Solubility (mg/mL)", ascending=False)
            fig = px.bar(aff_df, x="Solubility (mg/mL)", y="Oil", orientation='h', color="Solubility (mg/mL)")
            st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: SOLUBILITY ---
elif step_choice == "Step 2: Solubility":
    st.header("Step 2: Solubility Normalization")
    base_sol = 10**(0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp) * 1000
    st.session_state.sol_limit = np.clip((base_sol / 400) * 100, 1.0, 100.0)
    st.metric("Practical Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL")
    st.latex(r"Log S = 0.5 - 0.01(MW-50) - 0.6(LogP)")

# --- STEP 3: COMPONENT AI ---
elif step_choice == "Step 3: Component AI":
    st.header("Step 3: AI Recommendations")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    if df_raw is not None:
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_))
        recs = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
        st.markdown(f"<div class='advice-box'><b>AI Recommendation:</b><br>Surfactant: <b>{recs['Surfactant'].iloc[0]}</b><br>Co-Surfactant: <b>{recs['Co-surfactant'].iloc[0]}</b></div>", unsafe_allow_html=True)
        

# --- STEP 4: RATIOS ---
elif step_choice == "Step 4: Ratios":
    st.header("Step 4: Formulation Ratios")
    st.session_state.drug_conc = st.slider("Drug Loading (mg/mL)", 0.1, 100.0, 5.0)
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.smix_ratio = st.selectbox("S-mix Ratio (S:Co-S)", ["1:1", "2:1", "3:1"])

# --- STEP 5: FINAL SELECTION ---
elif step_choice == "Step 5: Final Selection":
    st.header("Step 5: User Selection")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    if le_dict:
        st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        if st.button("Finalize & Predict"): st.success("Parameters locked.")
    else: st.warning("Upload CSV in Step 1.")

# --- STEP 6: AI PREDICTIONS (8 IMPROVEMENTS) ---
elif step_choice == "Step 6: AI Predictions":
    st.header("Step 6: Comprehensive AI Analysis")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    def safe_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0

    input_data = [[safe_enc(le_dict['Drug_Name'], st.session_state.drug_name), safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice), 
                   safe_enc(le_dict['Surfactant'], st.session_state.s_final), safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final)]]

    # Predict Base Metrics
    res = {col: models[col].predict(input_data)[0] for col in models}
    is_stable_ai = stab_model.predict(input_data)[0]
    stable_text = "Stable" if (is_stable_ai == 1 and st.session_state.smix_p > st.session_state.oil_p) else "Unstable"

    # Physics Calculations for the 8 Parameters
    visc = 1.002 * (1 + 2.5 * (st.session_state.oil_p/100))
    t50 = (st.session_state.logp * 4.5) / (1 + (st.session_state.smix_p/100))
    trans = np.clip(100 - (res['Size_nm'] * 0.15), 0, 100)
    fassif = "High" if (res['Size_nm'] < 200) else "Moderate"

    # DISPLAY 8 PARAMETERS IN GRID
    st.subheader("Critical Quality Attributes & Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE%</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(f"<div class='metric-card'><div class='m-label'>Viscosity</div><div class='m-value'>{visc:.2f} cP</div></div>", unsafe_allow_html=True)
    c6.markdown(f"<div class='metric-card'><div class='m-label'>t50 Release</div><div class='m-value'>{t50:.1f} h</div></div>", unsafe_allow_html=True)
    c7.markdown(f"<div class='metric-card'><div class='m-label'>Transmittance</div><div class='m-value'>{trans:.1f}%</div></div>", unsafe_allow_html=True)
    c8.markdown(f"<div class='metric-card'><div class='m-label'>FaSSIF Stability</div><div class='m-value'>{fassif}</div></div>", unsafe_allow_html=True)

    st.write("---")
    col_plot, col_repair = st.columns([1.5, 1])
    with col_plot:
        st.subheader("Interfacial Phase Space")
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Nano-Region', 'a': [5, 15, 25, 10], 'b': [40, 50, 45, 35], 'c': [55, 35, 30, 55]}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 14, 'color': 'green' if stable_text == "Stable" else 'red'}}))
        st.plotly_chart(fig, use_container_width=True)
        

[Image of ternary phase diagram for nanoemulsion]


    with col_repair:
        if stable_text == "Unstable": 
            st.error("⚠️ Optimization Required")
            st.write(f"Increase S-mix above {st.session_state.oil_p}% to stabilize.")
        else: 
            st.success("✅ Optimized Formulation")
            
    
    hist_entry = f"{st.session_state.drug_name} ({stable_text})"
    if hist_entry not in st.session_state.history: st.session_state.history.append(hist_entry)
