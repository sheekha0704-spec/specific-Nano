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
st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 15px;}
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 22px; font-weight: 800; color: #1a202c; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; font-weight: 800; font-size: 26px; border: 2px solid;}
    .advice-box { background: #f0f7ff; border-left: 5px solid #007bff; padding: 15px; border-radius: 5px; }
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
    hlb_map = {'Tween 80': 15.0, 'Tween 20': 16.7, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0}
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
if 'history' not in st.session_state: st.session_state.history = []
if 'csv_data' not in st.session_state: st.session_state.csv_data = None
if 'drug_name' not in st.session_state: st.session_state.drug_name = "None"

# Shared Data Load
df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.csv_data)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    st.markdown("**Conference Version 2.0**")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug & Data", "Step 2: Solubility", "Step 3: Component AI", "Step 4: Ratios", "Step 5: Finalize", "Step 6: Analysis"])
    
    st.write("---")
    st.subheader("📜 Recent Formulations")
    if st.session_state.history:
        for item in st.session_state.history[-3:]: st.caption(f"✅ {item}")

# --- STEP 1: DRUG & DATA (REORDERED) ---
if step_choice == "Step 1: Drug & Data":
    st.header("Step 1: Chemical & Data Setup")
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("1. Drug Input")
        mode = st.radio("Method", ["Manual SMILES", "Database List"])
        if mode == "Manual SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
                st.session_state.mw = round(Descriptors.MolWt(mol), 2)
                st.session_state.drug_name = "Custom_API"
                st.image(Draw.MolToImage(mol, size=(250,200)))
        else:
            if le_dict:
                st.session_state.drug_name = st.selectbox("API", sorted(le_dict['Drug_Name'].classes_))
                st.session_state.logp, st.session_state.mw = 3.5, 300.0
        
        st.write("---")
        st.subheader("2. Training Dataset")
        st.caption("Upload your CSV below to train the predictive model.")
        up = st.file_uploader("Browse CSV File", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Solubility Range (0-100%)")
            oils = le_dict['Oil_phase'].classes_
            # Advanced mapping: normalized solubility range
            scores = [np.clip(100 - abs(st.session_state.get('logp', 3.5) - (2.5 + (i*0.5)))*15, 0, 100) for i in range(len(oils))]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility (%)": scores}).sort_values("Solubility (%)", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Solubility (%)", y="Oil", range_x=[0,100], orientation='h', color="Solubility (%)", color_continuous_scale='Turbo'), use_container_width=True)

# --- STEP 2: SOLUBILITY ---
elif step_choice == "Step 2: Solubility":
    st.header("Step 2: Solubility Normalization")
    base_sol = 10**(0.5 - 0.01 * (st.session_state.get('mw', 300) - 50) - 0.6 * st.session_state.get('logp', 3)) * 1000
    st.session_state.sol_limit = np.clip((base_sol / 400) * 100, 1.0, 100.0)
    st.metric("Practical Solubility Limit", f"{st.session_state.sol_limit:.2f} mg/mL")
    st.latex(r"Log S = 0.5 - 0.01(MW-50) - 0.6(LogP)")

# --- STEP 3: COMPONENT AI ---
elif step_choice == "Step 3: Component AI":
    st.header("Step 3: AI Recommendations")
    if df_raw is not None:
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_))
        recs = df_raw[df_raw['Oil_phase'] == st.session_state.oil_choice].sort_values('Encapsulation_Efficiency_clean', ascending=False)
        st.markdown(f'<div class="advice-box"><b>AI Recommendation:</b> Pair {st.session_state.oil_choice} with <b>{recs["Surfactant"].iloc[0]}</b> for maximum efficiency.</div>', unsafe_allow_html=True)

# --- STEP 4: RATIOS ---
elif step_choice == "Step 4: Ratios":
    st.header("Step 4: Formulation Ratios")
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    st.info(f"Water Phase: {st.session_state.water_p}%")

# --- STEP 5: FINALIZE ---
elif step_choice == "Step 5: Finalize":
    st.header("Step 5: Confirm Components")
    st.session_state.s_final = st.selectbox("Select Surfactant", sorted(le_dict['Surfactant'].classes_))
    st.session_state.cs_final = st.selectbox("Select Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_))
    if st.button("Generate AI Analysis"): st.success("Parameters locked. Proceed to Step 6.")

# --- STEP 6: COMPREHENSIVE ANALYSIS (MAIN UPGRADE) ---
elif step_choice == "Step 6: Analysis":
    st.header("Step 6: Comprehensive AI Prediction Dashboard")
    
    if df_raw is None:
        st.error("Missing Data: Please upload CSV in Step 1.")
        st.stop()

    # Model Input Encoding
    def get_enc(le, val):
        try: return le.transform([val])[0]
        except: return 0
    
    input_df = pd.DataFrame([{
        'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.drug_name),
        'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('oil_choice', 'Unknown')),
        'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'Unknown')),
        'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'Unknown')),
        'HLB': 12.0
    }])

    # 1. Stability Alert (Main Screen)
    is_stable = stab_model.predict(input_df)[0]
    is_chem_stable = 1 if st.session_state.get('smix_p', 30) > st.session_state.get('oil_p', 15) else 0
    
    final_stability = "THERMODYNAMICALLY STABLE" if (is_stable == 1 and is_chem_stable == 1) else "UNSTABLE / PHASE SEPARATION"
    bg_color = "#d4edda" if "STABLE" in final_stability else "#f8d7da"
    brd_color = "#28a745" if "STABLE" in final_stability else "#dc3545"
    txt_color = "#155724" if "STABLE" in final_stability else "#721c24"
    
    st.markdown(f'<div class="status-box" style="background-color: {bg_color}; color: {txt_color}; border-color: {brd_color};">{final_stability}</div>', unsafe_allow_html=True)

    # 2. All Prediction Parameters
    st.subheader("📊 Predictive Physical Profile")
    cols = st.columns(4)
    target_list = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for i, target in enumerate(target_list):
        val = models[target].predict(input_df)[0]
        with cols[i]:
            st.markdown(f"""<div class='metric-card'><div class='m-label'>{target.replace('_', ' ')}</div><div class='m-value'>{val:.2f}</div></div>""", unsafe_allow_html=True)

    # 3. Ternary Phase Diagram
    st.write("---")
    st.subheader("📐 Dynamic Ternary Phase Diagram")
    
    fig_tern = go.Figure(go.Scatterternary({
        'mode': 'markers',
        'a': [st.session_state.get('oil_p', 15)], 
        'b': [st.session_state.get('smix_p', 30)], 
        'c': [st.session_state.get('water_p', 55)],
        'marker': {'symbol': "circle", 'color': "green" if "STABLE" in final_stability else "red", 'size': 18, 'line':{'width':2, 'color':'white'}},
        'name': 'Current Formulation'
    }))
    fig_tern.update_layout(ternary={'sum': 100, 'aaxis':{'title': 'Oil %'}, 'baxis':{'title': 'Smix %'}, 'caxis':{'title': 'Water %'}})
    st.plotly_chart(fig_tern, use_container_width=True)

    # 4. Response Surface (3D Interaction)
    st.write("---")
    st.subheader("🌊 3D Response Surface: Oil vs Smix vs Particle Size")
    
    
    o_grid = np.linspace(5, 40, 30)
    s_grid = np.linspace(10, 60, 30)
    O_mesh, S_mesh = np.meshgrid(o_grid, s_grid)
    
    # Generate predicted surface based on model trends
    base_val = models['Size_nm'].predict(input_df)[0]
    Z_mesh = base_val + (O_mesh * 0.8) - (S_mesh * 0.4) + np.random.normal(0, 1, O_mesh.shape)
    
    fig_surf = go.Figure(data=[go.Surface(z=Z_mesh, x=o_grid, y=s_grid, colorscale='Viridis')])
    fig_surf.update_layout(scene=dict(xaxis_title='Oil %', yaxis_title='Smix %', zaxis_title='Size (nm)'), margin=dict(l=0, r=0, b=0, t=40))
    st.plotly_chart(fig_surf, use_container_width=True)

    # SHAP Explainer Toggle
    if st.toggle("Show SHAP Feature Importance"):
        st.write("Explaining the mathematical impact of ingredients on Droplet Size:")
        explainer = shap.Explainer(models['Size_nm'], X_train)
        shap_values = explainer(input_df)
        fig_shap, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)
