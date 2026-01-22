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
st.set_page_config(page_title="NanoPredict AI v33.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 10px; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 20px; font-weight: bold; height: 3em; background-color: #28a745; color: white; }
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
        else:
            df = pd.DataFrame({'Drug_Name':['A'], 'Surfactant':['S'], 'Co-surfactant':['CS'], 'Oil_phase':['O'], 
                               'Size_nm':[100], 'PDI':[0.1], 'Zeta_mV':[-20], 'Encapsulation_Efficiency':[90], 'Stability':['stable']})
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    le_dict = {}
    df_train = df.copy()
    for col in cat_cols:
        df_train[col] = df_train[col].fillna("Unknown").astype(str).str.strip()
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df_train[f'{col}_clean'] = df_train[col].apply(get_num)
    
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol} if mol else None
    except: return None

# --- INITIALIZE STATE ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'history' not in st.session_state: st.session_state.history = []
if 'logp' not in st.session_state: st.session_state.logp = 3.0
if 'mw' not in st.session_state: st.session_state.mw = 300.0

# --- NAVIGATION FUNCTIONS ---
def move_next(): st.session_state.step += 1
def move_back(): st.session_state.step -= 1
def reset_app(): st.session_state.step = 0

# --- LOAD DATA ---
df_raw, models, stab_model, le_dict = load_and_prep()

# --- SIDEBAR ---
with st.sidebar:
    st.title("SOP Control")
    st.progress((st.session_state.step + 1) / 6)
    if st.button("🔄 Reset Formulation"): reset_app()
    st.write("---")
    st.subheader("Step-wise History")
    for item in st.session_state.history[-5:]: st.caption(f"✓ {item}")

# --- APP WORKFLOW ---
steps = ["Molecules", "Solubility", "AI Match", "Ratios", "Verify", "Report"]

if st.session_state.step == 0:
    st.header("Step 1: Drug Structural Profiling")
    smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
    info = get_chem_info(smiles)
    if info:
        st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
        st.session_state.drug_name = "Custom_API"
        st.image(Draw.MolToImage(info['mol'], size=(300,300)))
        st.success(f"Properties: LogP={st.session_state.logp}, MW={st.session_state.mw} Da")
    st.button("Analyze Solubility ➡️", on_click=move_next)

elif st.session_state.step == 1:
    st.header("Step 2: Predictive Solubility Engine")
    log_s = 0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp
    st.session_state.sol_limit = np.clip((10**log_s) * st.session_state.mw * 1000, 0.5, 200.0)
    
    st.metric("Saturation Solubility", f"{st.session_state.sol_limit:.2f} mg/mL")
    st.latex(r"Log S = 0.5 - 0.01(MW-50) - 0.6(LogP)")
    
    
    
    st.button("Find Ideal Excipients ➡️", on_click=move_next)

elif st.session_state.step == 2:
    st.header("Step 3: Formulation Optimizer")
    st.session_state.oil_choice = st.selectbox("Select Target Lipid", sorted(le_dict['Oil_phase'].classes_))
    
    # Auto-Optimizer logic: Find components with highest predicted EE%
    matches = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
    
    st.markdown(f"""<div class="advice-box"><b>Optimizer Recommendation:</b><br>
    The AI suggests <b>{matches['Surfactant'].iloc[0]}</b> and <b>{matches['Co-surfactant'].iloc[0]}</b> for maximum efficiency.</div>""", unsafe_allow_html=True)
    
    
    
    st.button("Set Concentration Ratios ➡️", on_click=move_next)

elif st.session_state.step == 3:
    st.header("Step 4: Dynamic Ternary Mapping")
    st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    
    water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'size': 18, 'color': 'green' if st.session_state.smix_p > st.session_state.oil_p else 'red'}}))
    st.plotly_chart(fig, use_container_width=True)
    
    st.button("Finalize Components ➡️", on_click=move_next)

elif st.session_state.step == 4:
    st.header("Step 5: Confirmation")
    st.session_state.s_final = st.selectbox("Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    st.button("Generate Performance Report ➡️", on_click=move_next)

elif st.session_state.step == 5:
    st.header("Step 6: AI Predictions")
    def get_e(cat, val): return le_dict[cat].transform([val])[0] if val in le_dict[cat].classes_ else 0
    inp = [[0, get_e('Oil_phase', st.session_state.oil_choice), get_e('Surfactant', st.session_state.s_final), get_e('Co-surfactant', st.session_state.cs_final)]]
    
    res = {col: models[col].predict(inp)[0] for col in models}
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{res['Size_nm']:.1f}nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta</div><div class='m-value'>{res['Zeta_mV']:.1f}mV</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE%</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)

    

    if st.button("Finish & Log Result"):
        st.session_state.history.append(f"{st.session_state.oil_choice} formulation optimized.")
        reset_app()
        st.rerun()
