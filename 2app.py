import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v6.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 22px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 10px solid #28a745;
        margin-bottom: 20px;
    }
    .m-label { font-size: 14px; color: #555; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 26px; color: #000; font-weight: 800; white-space: nowrap; }
    .axis-box { background: #fdfefe; padding: 15px; border-radius: 10px; border: 1px solid #dcdde1; margin-bottom: 20px; }
    .rationale-text { border-left: 5px solid #3498db; padding-left: 15px; background: #f7fbff; padding: 15px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE DATA ENGINE (CRITICAL FIX FOR NaN ERROR) ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Please upload '{csv_file}' to your GitHub repo.")
        st.stop()
    
    df = pd.read_csv(csv_file)
    
    # Helper to force numeric conversion
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    
    # Clean the targets and drop rows where target is NaN
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        
    # STRICTOR CLEANING: Drop any row that has a NaN in any target column
    # This prevents the "Input y contains NaN" error
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets])
    
    # Label Encoding
    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col].astype(str))
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    # Train Models on the cleaned df_train
    models = {col: GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').astype(int).fillna(0)
    stab_model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. STRUCTURE PREDICTION ---
@st.cache_data
def get_structure(drug_name):
    if not HAS_CHEM_LIBS: return None, None, None, None
    try:
        comp = pcp.get_compounds(drug_name, 'name')[0]
        mol = Chem.MolFromSmiles(comp.canonical_smiles)
        return Draw.MolToImage(mol, size=(300, 300)), comp.canonical_smiles, comp.molecular_weight, comp.xlogp
    except: return None, None, None, None

# --- 3. UI LAYOUT ---
page = st.sidebar.radio("Navigation", ["Step 1: Chemical Data", "Step 2: AI Recommendation", "Step 3: Outcome Prediction"])

if 'state' not in st.session_state:
    st.session_state.state = {'drug': sorted(df['Drug_Name'].unique())[0], 'oil': sorted(df['Oil_phase'].unique())[0]}

# --- PAGE 1 ---
if page == "Step 1: Chemical Data":
    st.header("1. API Selection & Structural Prediction")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.state['drug'] = st.selectbox("API Name", sorted(df['Drug_Name'].unique()))
        st.session_state.state['oil'] = st.selectbox("Lipid/Oil Phase", sorted(df['Oil_phase'].unique()))
    with c2:
        img, smi, mw, lp = get_structure(st.session_state.state['drug'])
        if img:
            st.image(img, caption=f"Chemical Structure: {st.session_state.state['drug']}")
            st.write(f"**MW:** {mw} | **LogP:** {lp}")
        else:
            st.info("Structure visualization requires rdkit/pubchempy and packages.txt.")

# --- PAGE 2 ---
elif page == "Step 2: AI Recommendation":
    st.header("2. Expert Formulation Rationale")
    oil = st.session_state.state['oil']
    best = df[df['Oil_phase'] == oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False).iloc[0]
    
    st.markdown(f"### Specific Recommendation for **{oil}**")
    st.markdown(f"""
    <div class="rationale-text">
    <b>Data-Driven Rationale:</b> For this oil phase, the AI recommends a surfactant system of <b>{best['Surfactant']}</b>. 
    In your experimental dataset, this specific combination produced a droplet size of <b>{best['Size_nm_clean']:.2f} nm</b> 
    and an efficiency of <b>{best['Encapsulation_Efficiency_clean']:.1f}%</b>. This system provides the optimal HLB balance 
    to emulsify {oil} while maintaining the stability of {st.session_state.state['drug']}.
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 3 ---
elif page == "Step 3: Outcome Prediction":
    st.header("3. Multi-Output Prediction & Ternary Mapping")
    st.markdown("""
    <div class="axis-box">
    <b>Public Axis Guide:</b> <b>X:</b> Oil Phase Content | <b>Y:</b> S-mix (Surfactant+Co-S) | <b>Z:</b> Aqueous (Water) Phase
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        s = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
        cs = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Run Prediction Engine"):
            inputs = [[le_dict['Drug_Name'].transform([st.session_state.state['drug']])[0],
                       le_dict['Oil_phase'].transform([st.session_state.state['oil']])[0],
                       le_dict['Surfactant'].transform([s])[0],
                       le_dict['Co-surfactant'].transform([cs])[0]]]
            
            res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
            conf = stab_model.predict_proba(inputs)[0][1] * 100
            
            metrics = [("Droplet Size", f"{res[0]:.2f} nm"), ("PDI", f"{res[1]:.3f}"), ("Zeta", f"{res[2]:.1f} mV"),
                       ("Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability Confidence", f"{conf:.1f} %")]
            
            for l, v in metrics:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    with c2:
        o_v, s
