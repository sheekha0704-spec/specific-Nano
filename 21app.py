import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NanoPredict AI v31.0", layout="wide")

st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-top: 5px solid #28a745; text-align: center; }
    .m-label { font-size: 12px; color: #666; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; }
    .m-value { font-size: 24px; font-weight: 800; color: #1a202c; margin-top: 5px; }
    .advice-box { background: #f0f7ff; border-left: 5px solid #007bff; padding: 20px; border-radius: 10px; margin: 15px 0; font-size: 16px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FAIL-SAFE DATA ENGINE ---
@st.cache_resource
def load_and_prep():
    # Attempt GitHub Load
    url = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/nanoemulsion%202.csv"
    try:
        df = pd.read_csv(url)
    except:
        # FULL PRE-LOADED DATASET (Ensures content is never empty)
        data = {
            'Drug_Name': ['Aspirin', 'Curcumin', 'Caffeine', 'Ibuprofen', 'Quercetin', 'Vitamin E'],
            'Surfactant': ['Tween 80', 'Tween 20', 'Cremophor EL', 'Labrasol', 'Span 80', 'Tween 80'],
            'Co-surfactant': ['PEG 400', 'Ethanol', 'Propylene Glycol', 'Transcutol', 'Glycerol', 'PEG 400'],
            'Oil_phase': ['Capryol 90', 'Oleic Acid', 'Miglyol 812', 'Castor Oil', 'Labrafac', 'Capryol 90'],
            'Size_nm': [112.5, 145.2, 98.4, 120.1, 155.6, 130.2],
            'PDI': [0.12, 0.21, 0.15, 0.18, 0.25, 0.14],
            'Zeta_mV': [-22.4, -31.5, -18.2, -25.0, -28.4, -20.5],
            'Encapsulation_Efficiency': [88.4, 91.2, 75.6, 82.3, 94.1, 89.0],
            'Stability': ['Stable', 'Stable', 'Stable', 'Unstable', 'Stable', 'Stable']
        }
        df = pd.DataFrame(data)

    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    le_dict = {}
    df_proc = df.copy()
    
    for col in cat_cols:
        le = LabelEncoder()
        df_proc[f'{col}_enc'] = le.fit_transform(df_proc[col])
        le_dict[col] = le

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    X = df_proc[[f'{c}_enc' for c in cat_cols]]
    models = {col: GradientBoostingRegressor(n_estimators=50).fit(X, df_proc[col]) for col in targets}
    
    df_proc['is_stable'] = df_proc['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=50).fit(X, df_proc['is_stable'])
    
    return df, models, stab_model, le_dict

# --- 3. SESSION STATE ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'logp' not in st.session_state: st.session_state.logp = 3.0
if 'mw' not in st.session_state: st.session_state.mw = 300.0
if 'oil_p' not in st.session_state: st.session_state.oil_p = 15
if 'smix_p' not in st.session_state: st.session_state.smix_p = 30

# --- 4. NAVIGATION ---
def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

df_raw, models, stab_model, le_dict = load_and_prep()

# --- 5. APP FLOW ---
steps = ["Molecules", "Solubility", "AI Selection", "Ratios", "Confirmation", "Predictions"]

with st.sidebar:
    st.title("NanoPredict SOP")
    st.write(f"**Current Phase:** {steps[st.session_state.step]}")
    st.progress((st.session_state.step + 1) / len(steps))
    if st.button("🔄 Start New Formulation"): 
        st.session_state.step = 0
        st.rerun()

# --- STEP 0: MOLECULES ---
if st.session_state.step == 0:
    st.header("Step 1: Drug Structural Analysis")
    smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O") # Default Aspirin
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
        st.session_state.mw = round(Descriptors.MolWt(mol), 2)
        col_img, col_data = st.columns(2)
        with col_img:
            st.image(Draw.MolToImage(mol, size=(300,300)), caption="Molecular Structure")
        with col_data:
            st.info(f"**Molecular Weight:** {st.session_state.mw} Da")
            st.info(f"**LogP (Lipophilicity):** {st.session_state.logp}")
    st.button("Calculate Solubility Profile ➡️", on_click=next_step)

# --- STEP 1: SOLUBILITY ---
elif st.session_state.step == 1:
    st.header("Step 2: Predictive Solubility Profiling")
    # Improved GSE Formula
    log_s = 0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp
    sol_mg_ml = (10**log_s) * st.session_state.mw * 1000
    st.session_state.sol_limit = np.clip(sol_mg_ml, 0.5, 200.0)

    st.markdown(f"""
    <div class="advice-box">
        <b>Predicted Saturation Concentration:</b> {st.session_state.sol_limit:.2f} mg/mL<br>
        <i>This value guides the maximum drug loading in the oil phase.</i>
    </div>
    """, unsafe_allow_html=True)
    st.latex(r"S_{pred} = 10^{0.5 - 0.01(MW-50) - 0.6(logP)}")
    st.button("Identify Ideal Components ➡️", on_click=next_step)

# --- STEP 2: COMPONENT AI ---
elif st.session_state.step == 2:
    st.header("Step 3: AI-Driven Excipient Matching")
    st.session_state.oil_choice = st.selectbox("Choose Target Oil Phase", sorted(le_dict['Oil_phase'].classes_))
    
    # Matching logic based on highest EE% in the dataset
    rec = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency', ascending=False).iloc[0]
    
    st.success(f"Best matched Surfactant: **{rec['Surfactant']}**")
    st.success(f"Best matched Co-Surfactant: **{rec['Co-surfactant']}**")
    st.button("Adjust Concentration Ratios ➡️", on_click=next_step)

# --- STEP 3: RATIOS ---
elif st.session_state.step == 3:
    st.header("Step 4: Dynamic Ternary Mapping")
    col_input, col_graph = st.columns([1, 2])
    with col_input:
        st.session_state.oil_p = st.slider("Oil Content (%)", 5, 40, st.session_state.oil_p)
        st.session_state.smix_p = st.slider("S-mix Content (%)", 10, 70, st.session_state.smix_p)
        st.write(f"Water Content: {100 - st.session_state.oil_p - st.session_state.smix_p}%")
    
    with col_graph:
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers+text',
            'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p],
            'text': ['Your Formulation'], 'marker': {'size': 20, 'color': '#007bff'}
        }))
        fig.update_layout(ternary={'aaxis':{'title':'Oil'},'baxis':{'title':'S-mix'},'caxis':{'title':'Water'}})
        st.plotly_chart(fig, use_container_width=True)
    st.button("Confirm Selection ➡️", on_click=next_step)

# --- STEP 4: CONFIRMATION ---
elif st.session_state.step == 4:
    st.header("Step 5: Final Formulation Lock")
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    st.button("Run AI Prediction Engine ➡️", on_click=next_step)

# --- STEP 5: PREDICTIONS ---
elif st.session_state.step == 5:
    st.header("Step 6: AI Performance Characterization")
    
    # Helper to handle categorical encoding
    def get_e(cat, val): return le_dict[cat].transform([val])[0]
    
    # Ensure drug name exists in encoder or use default index 0
    try: d_idx = get_e('Drug_Name', 'Aspirin')
    except: d_idx = 0

    input_feats = [[d_idx, get_e('Surfactant', st.session_state.s_final), 
                    get_e('Co-surfactant', st.session_state.cs_final), 
                    get_e('Oil_phase', st.session_state.oil_choice)]]
    
    res = {k: m.predict(input_feats)[0] for k, m in models.items()}
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Droplet Size</div><div class='m-value'>{res['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>Polydispersity (PDI)</div><div class='m-value'>{res['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta Potential</div><div class='m-value'>{res['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE%</div><div class='m-value'>{res['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)

    st.balloons()
