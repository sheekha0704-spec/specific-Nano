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

# --- HELPER FUNCTIONS ---
def get_enc(le, val):
    try:
        return le.transform([val])[0]
    except:
        return 0

@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None, None, None, None, None
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): 
            df = pd.read_csv(csv_path)
        else: 
            return None, None, None, None, None

    hlb_map = {'Tween 80': 15.0, 'Tween 20': 16.7, 'Span 80': 4.3, 'Cremophor EL': 13.5, 'Labrasol': 12.0}
    df['HLB'] = df['Surfactant'].map(hlb_map).fillna(12.0)
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: 
        if col in df.columns:
            df[f'{col}_clean'] = df[col].apply(get_num)
        else:
            df[f'{col}_clean'] = 0.0
    
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

# Initializing Data
df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.get('csv_data'))

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug Selection & Range", "Step 2: Composition Selection", "Step 3: Ratio & Ternary Mapping", "Step 4: Final AI Predictions"])

# --- STEP 1 ---
if step_choice == "Step 1: Drug Selection & Range":
    st.header("Step 1: Drug & Multi-Component Solubility")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        method = st.radio("Selection Method", ["Manual Entry (Searchable)", "SMILES", "Browse CSV"])
        
        if method == "Manual Entry (Searchable)":
            if le_dict:
                st.session_state.drug_name = st.selectbox("Search Drug in Database", sorted(le_dict['Drug_Name'].classes_))
            else:
                st.session_state.drug_name = st.text_input("Drug Name", "Aspirin")
            st.session_state.logp, st.session_state.mw = 1.19, 180.16
            
        elif method == "SMILES":
            smiles = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp = round(Descriptors.MolLogP(mol), 2)
                st.session_state.mw = round(Descriptors.MolWt(mol), 2)
                st.image(Draw.MolToImage(mol, size=(200,150)))
                
        else:
            up = st.file_uploader("Upload Training CSV", type="csv")
            if up: 
                st.session_state.csv_data = up
                st.success("File synchronized! Dropdowns updated.")
                st.rerun()

    with c2:
        logp = st.session_state.get('logp', 3.0)
        st.subheader("Component Solubility Range (1-100%)")
        
        # Pulling list directly from CSV to ensure alignment
        oils = list(le_dict['Oil_phase'].classes_) if le_dict else ["MCT Oil", "Oleic Acid"]
        surfs = list(le_dict['Surfactant'].classes_) if le_dict else ["Tween 80", "Cremophor EL"]
        cosurfs = list(le_dict['Co-surfactant'].classes_) if le_dict else ["PEG 400", "Ethanol"]
        
        oil_scores = [np.clip(95 - abs(logp - (1.5 + i*0.5))*15, 5, 100) for i in range(len(oils))]
        surf_scores = [np.clip(90 - abs(logp - (0.5 + i*0.4))*12, 5, 100) for i in range(len(surfs))]
        cosurf_scores = [np.clip(85 - abs(logp - (0.2 + i*0.3))*10, 5, 100) for i in range(len(cosurfs))]
        
        range_df = pd.DataFrame({
            "Component": oils + surfs + cosurfs,
            "Solubility %": oil_scores + surf_scores + cosurf_scores,
            "Type": ["Oil"]*len(oils) + ["Surfactant"]*len(surfs) + ["Co-Surfactant"]*len(cosurfs)
        }).sort_values("Solubility %", ascending=False)
        st.plotly_chart(px.bar(range_df, x="Solubility %", y="Component", color="Type", orientation='h', range_x=[0,100], color_continuous_scale='Turbo'), use_container_width=True)

# --- STEP 2 ---
elif step_choice == "Step 2: Composition Selection":
    st.header("Step 2: Final Composition & Solubility")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.oil_final = st.selectbox("Select Oil", sorted(le_dict['Oil_phase'].classes_) if le_dict else ["None"])
        st.session_state.s_final = st.selectbox("Select Surfactant", sorted(le_dict['Surfactant'].classes_) if le_dict else ["None"])
        st.session_state.cs_final = st.selectbox("Select Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_) if le_dict else ["None"])
    
    with col2:
        st.subheader("Custom Solubility Parameters")
        logp = st.session_state.get('logp', 3.0)
        mw = st.session_state.get('mw', 300.0)
        sol_base = 10**(0.5 - 0.01 * (mw - 50) - 0.6 * logp) * 1000 
        
        st.session_state.sol_oil = st.number_input(f"Solubility in {st.session_state.oil_final} (mg/mL)", value=float(round((sol_base * (10**logp))/1000, 2)))
        st.session_state.sol_surf = st.number_input(f"Solubility in {st.session_state.s_final} (mg/L)", value=float(round(sol_base * 2.5, 4)))
        st.session_state.sol_cosurf = st.number_input(f"Solubility in {st.session_state.cs_final} (mg/L)", value=float(round(sol_base * 1.8, 4)))

# --- STEP 3 ---
elif step_choice == "Step 3: Ratio & Ternary Mapping":
    st.header("Step 3: Ratio Optimization & Ternary Diagram")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.smix_p = st.slider("S-mix %", 10, 70, 30)
        st.session_state.oil_p = st.slider("Oil %", 5, 50, 15)
        st.session_state.km_ratio = st.select_slider("Km (S:CoS Ratio)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        
    with col2:
        
        km_val = int(st.session_state.km_ratio.split(":")[0])
        base_bound = 15 + (km_val * 5)
        t_oil, t_smix = [0, 5, 15, base_bound, 0], [base_bound + 20, base_bound + 10, base_bound, 30, base_bound + 20]
        t_water = [100-x-y for x,y in zip(t_oil, t_smix)]

        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'a': t_oil, 'b': t_smix, 'c': t_water, 'fill': 'toself', 'fillcolor': 'rgba(40, 167, 69, 0.2)'}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p], 'marker': {'size': 18, 'color': 'red'}}))
        fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4 ---
elif step_choice == "Step 4: Final AI Predictions":
    st.header("Step 4: Characterization Predictions")
    if models:
        input_df = pd.DataFrame([{
            'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_name', 'Unknown')),
            'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('oil_final', 'None')),
            'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'None')),
            'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'None')),
            'HLB': 12.0
        }])
        res = {target: models[target].predict(input_df)[0] for target in models}
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Size", f"{res['Size_nm']:.2f} nm")
        p2.metric("PDI", f"{res['PDI']:.3f}")
        p3.metric("Zeta", f"{res['Zeta_mV']:.2f} mV")
        p4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}%")
        
        is_stable = stab_model.predict(input_df)[0]
        st.markdown(f'<div style="background-color: {"#d4edda" if is_stable else "#f8d7da"}; padding: 20px; border-radius: 10px; text-align: center;">{"STABLE" if is_stable else "UNSTABLE"}</div>', unsafe_allow_html=True)
