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
st.set_page_config(page_title="NanoPredict AI v30.0", layout="wide")

st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; }
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #eef6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (GITHUB AUTO-LOAD) ---
@st.cache_resource
def load_and_prep():
    # Direct Raw GitHub URL
    url = "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/nanoemulsion%202.csv"
    try:
        df = pd.read_csv(url)
    except:
        # Emergency Fallback if URL fails
        df = pd.DataFrame({
            'Drug_Name': ['Sample'], 'Surfactant': ['Tween 80'], 'Co-surfactant': ['PEG'],
            'Oil_phase': ['Capryol'], 'Size_nm': [150], 'PDI': [0.2], 'Zeta_mV': [-20],
            'Encapsulation_Efficiency': [90], 'Stability': ['Stable']
        })

    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    le_dict = {}
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col])
        le_dict[col] = le

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor().fit(X, df[col]) for col in targets}
    
    df['is_stable'] = df['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df['is_stable'])
    
    return df, models, stab_model, le_dict

# --- 3. SESSION STATE INITIALIZATION ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'history' not in st.session_state: st.session_state.history = []

# Core Logic Vars
defaults = {'logp': 3.0, 'mw': 300.0, 'oil_p': 15, 'smix_p': 30, 'drug_name': "None"}
for key, val in defaults.items():
    if key not in st.session_state: st.session_state[key] = val

# --- 4. NAVIGATION CONTROLS ---
steps = ["Step 1: Chemical Setup", "Step 2: Solubility", "Step 3: Component AI", 
         "Step 4: Ratios", "Step 5: Selection", "Step 6: Final Prediction"]

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

with st.sidebar:
    st.title("Navigation")
    st.progress((st.session_state.step + 1) / len(steps))
    for i, s in enumerate(steps):
        color = "✅" if st.session_state.step > i else "🔵" if st.session_state.step == i else "⚪"
        st.write(f"{color} {s}")
    if st.button("Reset Process"): st.session_state.step = 0

# --- 5. PAGE CONTENT ---
df_raw, models, stab_model, le_dict = load_and_prep()

# STEP 1: DRUG SETUP
if st.session_state.step == 0:
    st.header(steps[0])
    smiles = st.text_input("Enter Drug SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
        st.session_state.mw = round(Descriptors.MolWt(mol), 2)
        st.image(Draw.MolToImage(mol, size=(200,200)))
        st.success(f"Detected: {st.session_state.mw} g/mol | LogP: {st.session_state.logp}")
    st.button("Next: Analyze Solubility ➡️", on_click=next_step)

# STEP 2: DYNAMIC SOLUBILITY
elif st.session_state.step == 1:
    st.header(steps[1])
    # Custom formula to prevent the "1" default
    # Log S (Molar) = 0.5 - 0.01(MW-50) - 0.6(LogP)
    val = 0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp
    sol_mg_ml = (10**val) * st.session_state.mw * 1000 # Convert Molar to mg/mL
    st.session_state.sol_limit = np.clip(sol_mg_ml, 0.1, 150.0)

    st.metric("Custom Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL")
    st.latex(r"S_{calc} = 10^{0.5 - 0.01(MW-50) - 0.6(LogP)} \times MW \times 10^3")
    st.button("Next: AI Component Selection ➡️", on_click=next_step)

# STEP 3: COMPONENT AI
elif st.session_state.step == 2:
    st.header(steps[2])
    st.session_state.oil_choice = st.selectbox("Select Oil Phase", sorted(le_dict['Oil_phase'].classes_))
    # Recommendation based on EE%
    best = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency', ascending=False).iloc[0]
    st.markdown(f"""<div class="advice-box"><b>AI Recommendation for {st.session_state.oil_choice}:</b><br>
    Surfactant: {best['Surfactant']} | Co-Surfactant: {best['Co-surfactant']}</div>""", unsafe_allow_html=True)
    st.button("Next: Set Ratios ➡️", on_click=next_step)

# STEP 4: CUSTOM TERNARY RATIOS
elif st.session_state.step == 3:
    st.header(steps[3])
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, st.session_state.oil_p)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, st.session_state.smix_p)
    
    # Live Ternary Feedback
    water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    fig = go.Figure(go.Scatterternary({
        'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p],
        'marker': {'size': 15, 'color': 'blue'}
    }))
    fig.update_layout(ternary={'aaxis':{'title':'Oil'},'baxis':{'title':'S-mix'},'caxis':{'title':'Water'}})
    st.plotly_chart(fig)
    st.button("Next: Final Selection ➡️", on_click=next_step)

# STEP 5: SELECTION
elif st.session_state.step == 4:
    st.header(steps[4])
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    st.button("Next: Generate AI Predictions ➡️", on_click=next_step)

# STEP 6: FINAL ANALYSIS
elif st.session_state.step == 5:
    st.header(steps[5])
    def safe_enc(le, val): return le.transform([val])[0] if val in le.classes_ else 0
    
    input_data = [[0, safe_enc(le_dict['Oil_phase'], st.session_state.oil_choice), 
                    safe_enc(le_dict['Surfactant'], st.session_state.s_final), 
                    safe_enc(le_dict['Co-surfactant'], st.session_state.cs_final)]]
    
    preds = {col: models[col].predict(input_data)[0] for col in models}
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='m-label'>Size</div><div class='m-value'>{preds['Size_nm']:.1f} nm</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='m-label'>PDI</div><div class='m-value'>{preds['PDI']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='m-label'>Zeta</div><div class='m-value'>{preds['Zeta_mV']:.1f} mV</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='m-label'>EE%</div><div class='m-value'>{preds['Encapsulation_Efficiency']:.1f}%</div></div>", unsafe_allow_html=True)

    st.success("Analysis Complete. Use Sidebar to reset.")
