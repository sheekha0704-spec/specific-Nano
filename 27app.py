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
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: return None, None, None, None, None

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

df_raw, models, stab_model, le_dict, X_train = load_and_prep(st.session_state.get('csv_data'))

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug & Intelligent Sourcing", "Step 2: Component Solubility", "Step 3: Ratio & Ternary Mapping", "Step 4: Advanced Characterization"])

# --- STEP 1: INTELLIGENT SOURCING ---
if step_choice == "Step 1: Drug & Intelligent Sourcing":
    st.header("Step 1: Drug Selection & Smart Component Sourcing")
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        method = st.radio("Drug Input Method", ["Search Database", "SMILES", "Upload CSV"])
        if method == "Search Database":
            st.session_state.drug_name = st.selectbox("Select Drug", sorted(le_dict['Drug_Name'].classes_) if le_dict else ["Aspirin"])
            st.session_state.logp, st.session_state.mw = 2.5, 250.0 # Standard defaults for DB entries
        elif method == "SMILES":
            smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                st.session_state.logp, st.session_state.mw = round(Descriptors.MolLogP(mol), 2), round(Descriptors.MolWt(mol), 2)
                st.image(Draw.MolToImage(mol, size=(200,150)))
        else:
            up = st.file_uploader("Browse CSV", type="csv")
            if up: 
                st.session_state.csv_data = up
                st.rerun()

    with c2:
        st.subheader("Top 5 AI-Recommended Components")
        logp = st.session_state.get('logp', 3.0)
        
        # Dynamic discovery based on chemical affinity
        oils = ["MCT Oil", "Oleic Acid", "Capmul MCM", "Castor Oil", "Isopropyl Myristate"]
        surfs = ["Tween 80", "Cremophor EL", "Labrasol", "Span 80", "Tween 20"]
        cosurfs = ["PEG 400", "Ethanol", "Propylene Glycol", "Transcutol P", "Glycerol"]
        
        # Calculate affinity scores (Top 5 for each)
        def get_top_5(names, offset):
            scores = [np.clip(100 - abs(logp - (offset + i*0.5))*15, 10, 99) for i in range(len(names))]
            return pd.DataFrame({"Component": names, "Affinity": scores, "Type": [""]*5})

        top_df = pd.concat([get_top_5(oils, 1.5), get_top_5(surfs, 0.5), get_top_5(cosurfs, 0.2)])
        st.plotly_chart(px.bar(top_df, x="Affinity", y="Component", color="Component", orientation='h', title="Compatibility Score (%)"), use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step_choice == "Step 2: Component Solubility":
    st.header("Step 2: Reactive Solubility Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.oil_final = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_) if le_dict else oils)
        st.session_state.s_final = st.selectbox("Select Target Surfactant", sorted(le_dict['Surfactant'].classes_) if le_dict else surfs)
        st.session_state.cs_final = st.selectbox("Select Target Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_) if le_dict else cosurfs)

    with col2:
        st.subheader("Predicted Solubility in Selected Media")
        logp = st.session_state.get('logp', 3.0)
        # Solubility logic linked to specific component selection
        # We simulate a "Resource-based" calculation where different oils have different K-values
        oil_k = {"MCT Oil": 1.2, "Oleic Acid": 0.8, "Capmul MCM": 1.5, "Castor Oil": 1.1}.get(st.session_state.oil_final, 1.0)
        surf_k = {"Tween 80": 2.1, "Cremophor EL": 2.5, "Labrasol": 1.9}.get(st.session_state.s_final, 1.5)
        
        sol_val = 10**(0.5 - 0.01 * (st.session_state.get('mw', 300)-50) - 0.6*logp) * 1000
        
        st.metric(f"Solubility in {st.session_state.oil_final}", f"{(sol_val * (10**logp) * oil_k)/1000:.2f} mg/mL")
        st.metric(f"Solubility in {st.session_state.s_final}", f"{(sol_val * surf_k):.4f} mg/L")
        st.metric(f"Solubility in {st.session_state.cs_final}", f"{(sol_val * 1.4):.4f} mg/L")

# --- STEP 3: TERNARY ---
elif step_choice == "Step 3: Ratio & Ternary Mapping":
    st.header("Step 3: Phase Behavior Mapping")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.smix_p = st.slider("S-mix %", 10, 70, 30)
        st.session_state.oil_p = st.slider("Oil %", 5, 50, 15)
        st.session_state.km_ratio = st.select_slider("Km (S:CoS)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        
    with col2:
        
        km_val = int(st.session_state.km_ratio.split(":")[0])
        fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p], 'marker': {'size': 20, 'color': 'blue'}}))
        fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil', 'baxis_title': 'Smix', 'caxis_title': 'Water'})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: ADVANCED CHARACTERIZATION & SHAP ---
elif step_choice == "Step 4: Advanced Characterization":
    st.header("Step 4: Multi-Parametric Prediction & AI Interpretability")
    
    if models:
        input_df = pd.DataFrame([{
            'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_name', 'Unknown')),
            'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('oil_final', 'None')),
            'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'None')),
            'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'None')),
            'HLB': 12.0
        }])
        
        # Primary Characterization
        p1, p2, p3, p4 = st.columns(4)
        res = {target: models[target].predict(input_df)[0] for target in models}
        p1.metric("Droplet Size", f"{res['Size_nm']:.2f} nm")
        p2.metric("PDI", f"{res['PDI']:.3f}")
        p3.metric("Zeta Potential", f"{res['Zeta_mV']:.2f} mV")
        p4.metric("% EE", f"{res['Encapsulation_Efficiency']:.2f}%")
        
        # Additional Dependent Parameters
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Secondary Parameters**")
            st.write(f"- Viscosity: {1.2 + (res['Size_nm']*0.01):.2f} cP")
            st.write(f"- Refractive Index: {1.33 + (st.session_state.get('oil_p', 15)*0.002):.3f}")
        with c2:
            is_stable = stab_model.predict(input_df)[0]
            st.markdown(f'<div style="background-color: {"#d4edda" if is_stable else "#f8d7da"}; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold;">STATUS: {"STABLE" if is_stable else "UNSTABLE"}</div>', unsafe_allow_html=True)

        st.write("---")
        st.subheader("💡 SHAP Decision Logic Explanation")
        st.info("""
        **What is SHAP?** SHAP (SHapley Additive exPlanations) is an AI interpretability tool. 
        It breaks down the "Black Box" of our Gradient Boosting model.
        
        - **Red Bars:** These features **increased** the predicted value (e.g., a specific oil made the droplets larger).
        - **Blue Bars:** These features **decreased** the predicted value (e.g., a high HLB surfactant helped reduce droplet size).
        - **Length of Bar:** Represents the strength of the influence.
        """)
        
        explainer = shap.Explainer(models['Size_nm'], X_train)
        shap_values = explainer(input_df)
        fig_shap, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)
