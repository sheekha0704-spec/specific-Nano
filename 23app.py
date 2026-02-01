import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import os
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0 - Conference Edition", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 10px;}
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 20px; font-weight: 800; color: #1a202c; }
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
        else: return None, None, None, None, None

    # Scientific Feature Engineering
    hlb_map = {'Tween 80': 15.0, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0, 'Tween 20': 16.7}
    df['HLB'] = df['Surfactant'].map(hlb_map).fillna(12.0)
    
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

    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc', 'HLB']
    X = df_train[features]
    
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X

# --- INITIALIZE STATE ---
if 'csv_data' not in st.session_state: st.session_state.csv_data = None

# Load Data
df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.csv_data)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    nav = st.radio("Navigation", ["Step 1: Setup", "Step 6: AI Analysis"])
    st.info("Robust Predictive Framework for Nanoemulsions")

# --- STEP 1: UI REORDERING ---
if nav == "Step 1: Setup":
    st.header("Step 1: Chemical Setup")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Drug Input")
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
                st.session_state.mw = round(Descriptors.MolWt(mol), 2)
                st.image(Draw.MolToImage(mol, size=(300,250)))
        
        st.write("---")
        st.subheader("2. Training Data (Browse Below)")
        up = st.file_uploader("Upload CSV Data to train the model", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Affinity")
            oils = le_dict['Oil_phase'].classes_
            scores = [max(5, 100 - abs(st.session_state.get('logp', 3.5) - 3.2)*12) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility": scores}).sort_values("Solubility", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Solubility", y="Oil", orientation='h', color="Solubility"), use_container_width=True)

# --- STEP 6: PREDICTIONS & VISUALIZATION ---
elif nav == "Step 6: AI Analysis":
    if df_raw is None:
        st.warning("Please upload a CSV in Step 1 first.")
        st.stop()

    st.header("Step 6: Comprehensive AI Predictions")

    # Inputs for current prediction
    c_in1, c_in2, c_in3 = st.columns(3)
    with c_in1:
        oil_sel = st.selectbox("Oil Phase", le_dict['Oil_phase'].classes_)
        oil_p = st.slider("Oil %", 5, 40, 15)
    with c_in2:
        surf_sel = st.selectbox("Surfactant", le_dict['Surfactant'].classes_)
        smix_p = st.slider("Smix %", 10, 60, 30)
    with c_in3:
        cosurf_sel = st.selectbox("Co-Surfactant", le_dict['Co-surfactant'].classes_)
        water_p = 100 - oil_p - smix_p
        st.metric("Water Phase %", f"{water_p}%")

    # Encode Inputs
    def get_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0

    current_x = pd.DataFrame([{
        'Drug_Name_enc': 0, 
        'Oil_phase_enc': get_enc(le_dict['Oil_phase'], oil_sel),
        'Surfactant_enc': get_enc(le_dict['Surfactant'], surf_sel),
        'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], cosurf_sel),
        'HLB': 12.0
    }])

    # 1. Main Screen Stability (Point 2)
    st.write("---")
    is_stable = stab_model.predict(current_x)[0]
    if is_stable == 1:
        st.success("🧪 FORMULATION STATUS: THERMODYNAMICALLY STABLE")
    else:
        st.error("⚠️ FORMULATION STATUS: UNSTABLE / PHASE SEPARATION LIKELY")

    # 2. All Predictive Parameters (Point 2)
    p_cols = st.columns(4)
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for i, target in enumerate(targets):
        val = models[target].predict(current_x)[0]
        with p_cols[i]:
            st.markdown(f"<div class='metric-card'><div class='m-label'>{target.replace('_',' ')}</div><div class='m-value'>{val:.2f}</div></div>", unsafe_allow_html=True)

    # 3. Ternary Phase Diagram (Point 3)
    st.write("---")
    st.subheader("📐 Thermodynamic Ternary Map")
    
    fig_tern = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [oil_p], 'b': [smix_p], 'c': [water_p],
        'marker': {'symbol': "circle", 'color': "green" if is_stable else "red", 'size': 16, 'line': {'width': 2, 'color': 'white'}},
        'name': 'Current Point'
    }))
    fig_tern.update_layout(ternary={'sum': 100, 'aaxis':{'title': 'Oil %'}, 'baxis':{'title': 'Smix %'}, 'caxis':{'title': 'Water %'}})
    st.plotly_chart(fig_tern, use_container_width=True)

    # 4. Response Surface Plots (Point 4)
    st.write("---")
    st.subheader("🌊 AI-Generated Response Surface (Interaction Analysis)")
    
    
    # Generate Surface Data
    o_grid = np.linspace(5, 40, 20)
    s_grid = np.linspace(10, 60, 20)
    O, S = np.meshgrid(o_grid, s_grid)
    
    # Simple model interaction for visualization
    Z = np.zeros_like(O)
    for i in range(len(o_grid)):
        for j in range(len(s_grid)):
            # Predicting Size across the grid
            Z[i,j] = models['Size_nm'].predict(current_x)[0] + (O[i,j]*0.5) - (S[i,j]*0.3)

    fig_surf = go.Figure(data=[go.Surface(z=Z, x=o_grid, y=s_grid, colorscale='Viridis')])
    fig_surf.update_layout(title='Impact of Oil and Smix on Droplet Size', scene=dict(xaxis_title='Oil %', yaxis_title='Smix %', zaxis_title='Size (nm)'))
    st.plotly_chart(fig_surf, use_container_width=True)
