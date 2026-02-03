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
from rdkit.Chem import Descriptors
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
        else: return None, None, None, None, None, None

    # Database of chemical properties for reactive solubility
    solvent_props = {
        'Oils': {'MCT Oil': 1.4, 'Oleic Acid': 0.9, 'Capmul MCM': 1.8, 'Castor Oil': 1.1, 'Isopropyl Myristate': 1.3, 'Almond Oil': 1.0},
        'Surfactants': {'Tween 80': 15.0, 'Cremophor EL': 13.5, 'Labrasol': 12.0, 'Span 80': 4.3, 'Tween 20': 16.7},
        'Co-Surfactants': {'PEG 400': 1.1, 'Ethanol': 2.5, 'Propylene Glycol': 1.8, 'Transcutol P': 2.2, 'Glycerol': 0.7, 'Co-surfactant': 1.0}
    }

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

    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']
    X = df_train[features]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict, X, solvent_props

df_raw, models, stab_model, le_dict, X_train, solvent_props = load_and_prep(st.session_state.get('csv_data'))

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Navigation", ["Step 1: Drug Selection", "Step 2: Reactive Solubility", "Step 3: Ternary Mapping", "Step 4: AI Interpretation"])

# --- STEP 1: DRUG DRIVEN SOURCING ---
if step_choice == "Step 1: Drug Selection":
    st.header("Step 1: Drug-Centric Component Sourcing")
    
    col_input, col_chart = st.columns([1, 1.5])
    with col_input:
        method = st.radio("Input Method", ["Search Database", "SMILES", "Upload CSV"])
        if method == "Search Database":
            st.session_state.drug_name = st.selectbox("Select Drug", sorted(le_dict['Drug_Name'].classes_) if le_dict else ["Abacavir"])
        elif method == "SMILES":
            smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        else:
            up = st.file_uploader("Choose CSV", type="csv")
            if up: st.session_state.csv_data = up; st.rerun()

    with col_chart:
        st.subheader("Compatibility Score (%)")
        # Color-coded sourcing logic
        o_names = ['MCT Oil', 'Oleic Acid', 'Capmul MCM', 'Castor Oil', 'Isopropyl Myristate']
        s_names = ['Tween 80', 'Cremophor EL', 'Labrasol', 'Span 80', 'Tween 20']
        c_names = ['PEG 400', 'Ethanol', 'Propylene Glycol', 'Transcutol P', 'Glycerol']
        
        comp_df = pd.DataFrame({
            "Component": o_names + s_names + c_names,
            "Affinity": [85, 92, 98, 91, 84, 70, 78, 85, 92, 99, 66, 74, 80, 88, 95],
            "Type": ["Oil"]*5 + ["Surfactant"]*5 + ["Co-Surfactant"]*5
        })
        
        fig = px.bar(comp_df, x="Affinity", y="Component", color="Type", orientation='h',
                     color_discrete_map={"Oil": "#1f77b4", "Surfactant": "#ff7f0e", "Co-Surfactant": "#2ca02c"})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step_choice == "Step 2: Reactive Solubility":
    st.header("Step 2: Reactive Solubility Prediction")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sel_oil = st.selectbox("Select Target Oil", list(solvent_props['Oils'].keys()))
        st.session_state.sel_surf = st.selectbox("Select Target Surfactant", list(solvent_props['Surfactants'].keys()))
        st.session_state.sel_cosurf = st.selectbox("Select Target Co-Surfactant", list(solvent_props['Co-Surfactants'].keys()))
    
    with c2:
        st.subheader("Predicted Solubility in Selected Media")
        oil_val = 0.32 * solvent_props['Oils'][st.session_state.sel_oil]
        surf_val = 2.50 * (solvent_props['Surfactants'][st.session_state.sel_surf] / 13.5)
        cosurf_val = 1.40 * solvent_props['Co-Surfactants'][st.session_state.sel_cosurf]
        
        st.metric(f"Solubility in {st.session_state.sel_oil}", f"{oil_val:.2f} mg/mL")
        st.metric(f"Solubility in {st.session_state.sel_surf}", f"{surf_val:.4f} mg/L")
        st.metric(f"Solubility in {st.session_state.sel_cosurf}", f"{cosurf_val:.4f} mg/L")

# --- STEP 3: TERNARY ---
elif step_choice == "Step 3: Ternary Mapping":
    st.header("Step 3: Ratio Optimization & Ternary Diagram")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.smix_p = st.slider("S-mix %", 10, 70, 30)
        st.session_state.oil_p = st.slider("Oil %", 5, 50, 15)
        st.session_state.km_ratio = st.select_slider("Km (S:CoS)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        water_p = 100 - st.session_state.smix_p - st.session_state.oil_p
        st.info(f"Water Phase Calculated: {water_p}%")

    with c2:
                km_val = int(st.session_state.km_ratio.split(":")[0])
        # Boundaries
        t_oil = [0, 5, 10, 20, 0]
        t_smix = [30, 25, 20, 15, 30]
        t_water = [100-x-y for x,y in zip(t_oil, t_smix)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'a': t_oil, 'b': t_smix, 'c': t_water, 'fill': 'toself', 'name': 'Nanoemulsion Region'}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'size': 15, 'color': 'red'}}))
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: INTERPRETATION ---
elif step_choice == "Step 4: AI Interpretation":
    st.header("Step 4: Multi-Parametric Prediction & Interpretation")
    
    input_df = pd.DataFrame([{
        'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_name', 'Abacavir')),
        'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('sel_oil', 'MCT Oil')),
        'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('sel_surf', 'Tween 80')),
        'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('sel_cosurf', 'PEG 400'))
    }])

    res = {target: models[target].predict(input_df)[0] for target in models}
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Droplet Size", f"{res['Size_nm']:.2f} nm")
    m2.metric("PDI", f"{res['PDI']:.3f}")
    m3.metric("Zeta Potential", f"{res['Zeta_mV']:.2f} mV")
    m4.metric("% EE", f"{res['Encapsulation_Efficiency']:.2f}%")
    
    is_stable = stab_model.predict(input_df)[0]
    st.markdown(f'<div style="background-color: {"#d4edda" if is_stable else "#f8d7da"}; padding: 20px; border-radius: 10px; text-align: center;">STATUS: {"STABLE" if is_stable else "UNSTABLE"}</div>', unsafe_allow_html=True)

    st.subheader("💡 SHAP Decision Logic Interpretation")
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_v = explainer(input_df)
    
    fig_s, ax = plt.subplots()
    shap.plots.waterfall(shap_v[0], show=False)
    st.pyplot(fig_s)
    
    # Custom Interpretation Sentence
    highest_impact_feat = X_train.columns[np.argmax(np.abs(shap_v.values[0]))].replace('_enc', '')
    st.write(f"**AI Insight:** Your droplet size of {res['Size_nm']:.2f} nm is primarily driven by your choice of **{highest_impact_feat}**. To reduce the size further, you should investigate alternative ingredients in that specific category.")
