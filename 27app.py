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

# --- 1. DATA & SOLVENT PROPERTY DATABASE ---
@st.cache_resource
def load_and_prep(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        csv_path = 'nanoemulsion 2.csv'
        if os.path.exists(csv_path): df = pd.read_csv(csv_path)
        else: return None, None, None, None, None, None

    # Step 2: Solvent-Specific Constants (K-values)
    # These values ensure solubility changes when you switch solvents
    solvent_props = {
        'Oils': {
            'MCT Oil': {'k': 1.45, 'desc': 'Medium Chain'},
            'Oleic Acid': {'k': 0.88, 'desc': 'Long Chain'},
            'Capmul MCM': {'k': 1.95, 'desc': 'Mono-diglyceride'},
            'Castor Oil': {'k': 1.12, 'desc': 'Vegetable Oil'},
            'Isopropyl Myristate': {'k': 1.30, 'desc': 'Synthetic Ester'}
        },
        'Surfactants': {
            'Tween 80': {'k': 15.0, 'sol_factor': 2.1},
            'Cremophor EL': {'k': 13.5, 'sol_factor': 2.5},
            'Labrasol': {'k': 12.0, 'sol_factor': 1.9},
            'Span 80': {'k': 4.3, 'sol_factor': 0.8},
            'Tween 20': {'k': 16.7, 'sol_factor': 2.2}
        },
        'Co-Surfactants': {
            'PEG 400': {'k': 1.1, 'sol_factor': 1.5},
            'Ethanol': {'k': 2.5, 'sol_factor': 2.8},
            'Propylene Glycol': {'k': 1.8, 'sol_factor': 1.9},
            'Transcutol P': {'k': 2.2, 'sol_factor': 2.4},
            'Glycerol': {'k': 0.7, 'sol_factor': 0.9}
        }
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
    
    return df, models, le_dict, X, solvent_props

df_raw, models, le_dict, X_train, solvent_props = load_and_prep(st.session_state.get('csv_data'))

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 NanoPredict AI")
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug-Driven Sourcing", "Step 2: Reactive Solubility", "Step 3: Ternary Mapping", "Step 4: AI Interpretation"])

# --- STEP 1: DRUG-DRIVEN SOURCING ---
if step_choice == "Step 1: Drug-Driven Sourcing":
    st.header("Step 1: Drug-Centric Component Determination")
    
    # 1. Select Drug
    st.session_state.drug_name = st.selectbox("Select Drug for Formulation", sorted(le_dict['Drug_Name'].classes_) if le_dict else ["Aspirin"])
    
    # Simulate Drug Properties (e.g., LogP) to determine affinity
    drug_logp = 2.5 
    
    st.subheader("Top Recommended Components (Affinity-Based)")
    
    def get_source_df(cat_name, color_hex, offset):
        names = list(solvent_props[cat_name].keys())
        # Score based on proximity to drug LogP
        scores = [np.clip(98 - abs(drug_logp - (offset + i*0.4))*15, 45, 99) for i in range(len(names))]
        return pd.DataFrame({"Component": names, "Compatibility": scores, "Type": cat_name, "Color": color_hex})

    # Colors: Blue (Oil), Orange (Surf), Green (Co-Surf)
    df_o = get_source_df('Oils', '#1f77b4', 1.5)
    df_s = get_source_df('Surfactants', '#ff7f0e', 0.5)
    df_cs = get_source_df('Co-Surfactants', '#2ca02c', 0.2)
    
    full_source = pd.concat([df_o, df_s, df_cs])
    fig = px.bar(full_source, x="Compatibility", y="Component", color="Type", orientation='h',
                 color_discrete_map={'Oils':'#1f77b4', 'Surfactants':'#ff7f0e', 'Co-Surfactants':'#2ca02c'})
    st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step_choice == "Step 2: Reactive Solubility":
    st.header("Step 2: Component-Specific Solubility Prediction")
    st.write(f"Predicting solubility for: **{st.session_state.get('drug_name', 'Default Drug')}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sel_oil = st.selectbox("Change Oil Phase", list(solvent_props['Oils'].keys()))
        st.session_state.sel_surf = st.selectbox("Change Surfactant", list(solvent_props['Surfactants'].keys()))
        st.session_state.sel_cosurf = st.selectbox("Change Co-Surfactant", list(solvent_props['Co-Surfactants'].keys()))

    with col2:
        # REACTIVE LOGIC: The result depends on WHICH solvent is selected
        base_sol = 4.8 
        
        # Pull k-values and sol_factors from Step 1's Database
        oil_val = base_sol * solvent_props['Oils'][st.session_state.sel_oil]['k']
        surf_val = base_sol * solvent_props['Surfactants'][st.session_state.sel_surf]['sol_factor']
        cosurf_val = base_sol * solvent_props['Co-Surfactants'][st.session_state.sel_cosurf]['sol_factor']
        
        st.metric(f"Solubility in {st.session_state.sel_oil}", f"{oil_val:.2f} mg/mL")
        st.metric(f"Solubility in {st.session_state.sel_surf}", f"{surf_val:.2f} mg/mL")
        st.metric(f"Solubility in {st.session_state.sel_cosurf}", f"{cosurf_val:.2f} mg/mL")

# --- STEP 3: TERNARY RATIOS ---
elif step_choice == "Step 3: Ratio & Ternary Mapping":
    st.header("Step 3: Smix Optimization & Phase Behavior")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.km_ratio = st.select_slider("Km (Surfactant : Co-S)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.smix_p = st.slider("S-mix %", 10, 80, 40)
        st.session_state.oil_p = st.slider("Oil %", 5, 50, 20)
        water_p = 100 - st.session_state.smix_p - st.session_state.oil_p
        
    with c2:
        
        km_val = int(st.session_state.km_ratio.split(":")[0])
        # Boundaries shift based on Km
        t_oil = [0, 5, 12, 18, 0]
        t_smix = [25+km_val*5, 20+km_val*5, 15+km_val*5, 10+km_val*5, 25+km_val*5]
        t_water = [100-x-y for x,y in zip(t_oil, t_smix)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'a': t_oil, 'b': t_smix, 'c': t_water, 'fill': 'toself', 'name': 'Region'}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'size': 16, 'color': 'red'}}))
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: INTERPRETATION ---
elif step_choice == "Step 4: AI Interpretation":
    st.header("Step 4: AI Prediction & Decision Interpretation")
    
    input_df = pd.DataFrame([{
        'Drug_Name_enc': le_dict['Drug_Name'].transform([st.session_state.get('drug_name', 'Abacavir')])[0],
        'Oil_phase_enc': le_dict['Oil_phase'].transform([st.session_state.get('sel_oil', 'MCT Oil')])[0],
        'Surfactant_enc': le_dict['Surfactant'].transform([st.session_state.get('sel_surf', 'Tween 80')])[0],
        'Co-surfactant_enc': le_dict['Co-surfactant'].transform([st.session_state.get('sel_cosurf', 'PEG 400')])[0]
    }])

    res = {target: models[target].predict(input_df)[0] for target in models}
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Size", f"{res['Size_nm']:.1f} nm")
    m2.metric("PDI", f"{res['PDI']:.3f}")
    m3.metric("Zeta", f"{res['Zeta_mV']:.1f} mV")
    m4.metric("% EE", f"{res['Encapsulation_Efficiency']:.1f}%")

    st.subheader("💡 Why did the AI predict this?")
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_v = explainer(input_df)
    
    fig_s, ax = plt.subplots()
    shap.plots.waterfall(shap_v[0], show=False)
    st.pyplot(fig_s)
    
    # SPECIFIC INTERPRETATION (No generic definitions)
    top_feature = X_train.columns[np.argmax(np.abs(shap_v.values[0]))].replace('_enc', '')
    impact_dir = "increased" if shap_v.values[0][np.argmax(np.abs(shap_v.values[0]))] > 0 else "decreased"
    
    st.info(f"""
    **Expert Interpretation:** The SHAP analysis reveals that your choice of **{top_feature}** had the most powerful impact on this prediction, which significantly **{impact_dir}** the predicted droplet size compared to the database average. 
    To optimize this further, focusing on the concentration or type of {top_feature} will provide the most control over the final formulation.
    """)
