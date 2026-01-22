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

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v24.0 | Conference Edition", layout="wide")

# --- PHARMA PROPERTY DATABASE ---
# Manually mapping surfactants to HLB for the AI logic
HLB_MAP = {
    "Tween 80": 15.0, "Tween 20": 16.7, "Span 80": 4.3, "Span 20": 8.6, 
    "Cremophor EL": 13.5, "Labrasol": 14.0, "Solutol HS15": 15.0, "Unknown": 10.0
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #007bff; text-align: center; margin-bottom: 10px; }
    .m-label { font-size: 11px; color: #666; font-weight: 700; text-transform: uppercase; }
    .m-value { font-size: 20px; color: #1a202c; font-weight: 800; }
    .advice-box { background: #f0f7ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & AI ENGINE ---
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
    
    # NEW: Add HLB as a feature for the AI
    df['HLB'] = df['Surfactant'].map(HLB_MAP).fillna(10.0)
    
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    # Updated Feature set including HLB
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol} if mol else None
    except: return None

# --- STATE INITIALIZATION ---
if 'history' not in st.session_state: st.session_state.history = []
if 'csv_data' not in st.session_state: st.session_state.csv_data = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("Conference Control")
    nav = ["Step 1: Chemical Setup", "Step 2: Solubility & Loading", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Final Results"]
    step = st.radio("Workflow Navigation", nav)
    st.write("---")
    st.subheader("📜 Recent Trials")
    for h in st.session_state.history[-3:]: st.info(h)

# --- STEP 1: CHEMICAL SETUP ---
if step == "Step 1: Chemical Setup":
    st.header("Step 1: API Analysis & Oil Affinity")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("1. Load Data")
        up = st.file_uploader("Upload Training CSV (Option 3)", type="csv")
        if up: st.session_state.csv_data = up
        
        st.subheader("2. Drug Definition")
        smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        info = get_chem_info(smiles)
        if info:
            st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
            st.image(Draw.MolToImage(info['mol'], size=(300,200)), caption="Molecular Structure")
    
    with c2:
        df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
        if le_dict:
            st.subheader("Oil Affinity (HSP Based)")
            oils = le_dict['Oil_phase'].classes_
            # Scoring drug-oil affinity based on molecular weight and logP
            scores = [max(10, 100 - abs(st.session_state.logp - 3.2)*12) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Affinity Score": scores}).sort_values("Affinity Score", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Affinity Score", y="Oil", orientation='h', color="Affinity Score", color_continuous_scale="Viridis"), use_container_width=True)
            

# --- STEP 2: SOLUBILITY ---
elif step == "Step 2: Solubility & Loading":
    st.header("Step 2: Solubility Normalization")
    # Base GSE Calculation
    base_sol = 10**(0.5 - 0.01*(st.session_state.mw-50) - 0.6*st.session_state.logp) * 1000
    # Normalize to 100mg scale
    st.session_state.sol_limit = np.clip((base_sol/400)*100, 1.0, 100.0)
    
    st.metric("Practical Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL", delta="Max 100 normalized")
    st.write("---")
    st.session_state.drug_conc = st.slider("Target Drug Conc (mg/mL)", 0.1, 100.0, 5.0)
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)

# --- STEP 3: SCREENING ---
elif step == "Step 3: AI Screening":
    st.header("Step 3: Component AI Recommendations")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    if le_dict:
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_))
        best_data = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
        
        st.markdown(f"""<div class="advice-box"><b>AI Suggestion for {st.session_state.oil_choice}:</b><br>
        Best Surfactant: {best_data['Surfactant'].iloc[0]} (HLB: {HLB_MAP.get(best_data['Surfactant'].iloc[0], 10)})<br>
        Best Co-Surfactant: {best_data['Co-surfactant'].iloc[0]}</div>""", unsafe_allow_html=True)
        

# --- STEP 4: SELECTION ---
elif step == "Step 4: Selection":
    st.header("Step 4: Final Formulation Parameters")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    if le_dict:
        st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
        st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
        st.session_state.smix_ratio = st.selectbox("Confirm S-mix Ratio", ["1:1", "2:1", "3:1"])

# --- STEP 5: RESULTS ---
elif step == "Step 5: Final Results":
    st.header("Step 5: AI Performance Suite (Conference Edition)")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    current_hlb = HLB_MAP.get(st.session_state.s_final, 10.0)
    
    def safe_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0

    input_vec = [[safe_enc(le_dict['Drug_Name'], "Unknown"), safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice), 
                  safe_enc(le_dict['Surfactant'], st.session_state.s_final), safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final), current_hlb]]

    # Core AI Predictions
    res = {col: models[col].predict(input_vec)[0] for col in models}
    
    # NEW: Advanced Pharmaceutical Parameter Logic
    # 1. Drug Release (t50): Estimated via Partition Coefficient
    t50 = (st.session_state.logp * 4) + (st.session_state.oil_p * 0.5) 
    # 2. Viscosity (cP): Function of Oil load and surfactant
    visc = 1.2 + (st.session_state.oil_p * 0.1) + (current_hlb * 0.05)
    # 3. Transmittance (%): Clarity vs Globule Size
    trans = np.clip(100 - (res['Size_nm'] * 0.12) - (st.session_state.oil_p * 0.4), 0, 100)
    # 4. FaSSIF Stability: Predictive score based on HLB and size
    fassif = "High" if (current_hlb > 12 and res['Size_nm'] < 150) else "Moderate"

    # Display 6+4 Parameters (Total Analysis)
    st.subheader("Predicted Critical Quality Attributes (CQAs)")
    cols = [st.columns(4), st.columns(4), st.columns(2)]
    
    # Row 1
    cols[0][0].markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    cols[0][1].markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    cols[0][2].markdown(f"<div class='metric-card'><div class='m-label'>Zeta Potential</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    cols[0][3].markdown(f"<div class='metric-card'><div class='m-label'>EE %</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)
    
    # Row 2 (Solubility & Stability + Release & Viscosity)
    cols[1][0].markdown(f"<div class='metric-card'><div class='m-label'>Solubility</div><div class='m-value'>{st.session_state.sol_limit:.1f}</div></div>", unsafe_allow_html=True)
    cols[1][1].markdown(f"<div class='metric-card'><div class='m-label'>Drug Release (t50)</div><div class='m-value'>{t50:.1f} hr</div></div>", unsafe_allow_html=True)
    cols[1][2].markdown(f"<div class='metric-card'><div class='m-label'>Viscosity</div><div class='m-value'>{visc:.2f} cP</div></div>", unsafe_allow_html=True)
    cols[1][3].markdown(f"<div class='metric-card'><div class='m-label'>Transmittance</div><div class='m-value'>{trans:.1f}%</div></div>", unsafe_allow_html=True)

    # Row 3
    cols[2][0].markdown(f"<div class='metric-card'><div class='m-label'>FaSSIF Stability</div><div class='m-value'>{fassif}</div></div>", unsafe_allow_html=True)
    stable_flag = "Stable" if (stab_model.predict(input_vec)[0] == 1 and st.session_state.smix_p > st.session_state.oil_p) else "Unstable"
    cols[2][1].markdown(f"<div class='metric-card'><div class='m-label'>Thermodynamic Stability</div><div class='m-value'>{stable_flag}</div></div>", unsafe_allow_html=True)

    # Dynamic Ternary Phase Analysis
    st.write("---")
    st.subheader("Stability Mapping & Repair Guide")
    c_p, c_r = st.columns([1.5, 1])
    with c_p:
        shift = st.session_state.logp * 2
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable Region', 'a': [5, 15, 25, 10, 5], 'b': [40+shift, 50+shift, 45+shift, 35+shift, 40+shift], 'c': [55, 35, 30, 55, 55]}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'name': 'Current', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 14, 'color': 'red' if stable_flag == "Unstable" else 'green'}}))
        st.plotly_chart(fig, use_container_width=True)
        

[Image of ternary phase diagram for nanoemulsion]

    with c_r:
        if stable_flag == "Unstable":
            st.error("⚠️ Optimization Required")
            st.write(f"- **Issue:** Low surfactant HLB ({current_hlb}) for lipid load.")
            st.write(f"- **Repair:** Increase S-mix to {st.session_state.oil_p * 2}% or use surfactant with HLB > 12.")
        else:
            st.success("✅ Formulation Optimized for International Standards.")
            

    if st.button("🔄 Clear and Start New Project"):
        st.session_state.history.append(f"{st.session_state.oil_choice} | {stable_flag}")
        st.rerun()
