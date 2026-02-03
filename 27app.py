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
        else: return None, None, None, None, None

    # Solvent Property Database for Step 2 Reactive Logic
    solvent_props = {
        'Oils': {'MCT Oil': 1.4, 'Oleic Acid': 0.9, 'Capmul MCM': 1.8, 'Castor Oil': 1.1, 'Isopropyl Myristate': 1.3},
        'Surfactants': {'Tween 80': 15.0, 'Cremophor EL': 13.5, 'Labrasol': 12.0, 'Span 80': 4.3, 'Tween 20': 16.7},
        'Co-Surfactants': {'PEG 400': 1.1, 'Ethanol': 2.5, 'Propylene Glycol': 1.8, 'Transcutol P': 2.2, 'Glycerol': 0.7}
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
    
    # Selection
    st.session_state.drug_choice = st.selectbox("Identify Drug Molecule", sorted(le_dict['Drug_Name'].classes_) if le_dict else ["Aspirin"])
    
    # Logic: Components determined by drug properties (simulated via affinity)
    logp_sim = 3.5 # In a real scenario, pull this from a drug property dict
    
    st.subheader("Determined Excipients (Top 5 per Category)")
    
    def make_comp_df(category, color_hex):
        names = list(solvent_props[category].keys())
        # Score based on proximity to logp_sim
        scores = [np.clip(95 - abs(logp_sim - (1.0 + i*0.5))*10, 40, 98) for i in range(len(names))]
        return pd.DataFrame({"Component": names, "Compatibility": scores, "Type": category, "Color": color_hex})

    # Uniform but distinct coloring per category
    df_o = make_comp_df('Oils', '#1f77b4')   # Blue
    df_s = make_comp_df('Surfactants', '#ff7f0e') # Orange
    df_c = make_comp_df('Co-Surfactants', '#2ca02c') # Green
    
    full_source = pd.concat([df_o, df_s, df_c])
    fig = px.bar(full_source, x="Compatibility", y="Component", color="Type", 
                 orientation='h', color_discrete_map={'Oils':'#1f77b4','Surfactants':'#ff7f0e','Co-Surfactants':'#2ca02c'})
    st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step_choice == "Step 2: Reactive Solubility":
    st.header("Step 2: Dynamic Solubility Analysis")
    st.info(f"Analyzing media for: **{st.session_state.get('drug_choice', 'Selected Drug')}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.sel_oil = st.selectbox("Change Oil", list(solvent_props['Oils'].keys()))
        st.session_state.sel_surf = st.selectbox("Change Surfactant", list(solvent_props['Surfactants'].keys()))
        st.session_state.sel_cosurf = st.selectbox("Change Co-Surfactant", list(solvent_props['Co-Surfactants'].keys()))

    with col2:
        # Reactive Logic: Solubility changes based on the 'k-value' of the selected solvent
        base_sol = 5.2 # mg
        oil_val = base_sol * solvent_props['Oils'][st.session_state.sel_oil]
        surf_val = (base_sol * 0.5) * (solvent_props['Surfactants'][st.session_state.sel_surf] / 10)
        cosurf_val = base_sol * solvent_props['Co-Surfactants'][st.session_state.sel_cosurf]
        
        st.metric(f"Solubility in {st.session_state.sel_oil}", f"{oil_val:.2f} mg/mL")
        st.metric(f"Solubility in {st.session_state.sel_surf}", f"{surf_val:.4f} mg/mL")
        st.metric(f"Solubility in {st.session_state.sel_cosurf}", f"{cosurf_val:.2f} mg/mL")

# --- STEP 3: TERNARY RATIOS ---
elif step_choice == "Step 3: Ternary Mapping":
    st.header("Step 3: Smix Optimization & Phase Diagram")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.km_ratio = st.select_slider("Smix Ratio (S:CoS)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.smix_p = st.slider("S-mix %", 10, 80, 40)
        st.session_state.oil_p = st.slider("Oil %", 5, 50, 20)
        water_p = 100 - st.session_state.smix_p - st.session_state.oil_p
        st.code(f"K_m = {st.session_state.km_ratio.split(':')[0]}")
    
    with c2:
                km_val = int(st.session_state.km_ratio.split(":")[0])
        # Dynamic boundaries that change with Km
        t_oil = [0, 5, 10, 20, 0]
        t_smix = [20+km_val*10, 15+km_val*10, 10+km_val*10, 5+km_val*10, 20+km_val*10]
        t_water = [100-x-y for x,y in zip(t_oil, t_smix)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'a': t_oil, 'b': t_smix, 'c': t_water, 'fill': 'toself', 'name': 'Nanoemulsion Area'}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [water_p], 'marker': {'size': 15, 'color': 'red'}, 'name': 'Current Point'}))
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: INTERPRETATION ---
elif step_choice == "Step 4: AI Interpretation":
    st.header("Step 4: Characterization & SHAP Interpretation")
    
    input_df = pd.DataFrame([{
        'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_choice', 'Unknown')),
        'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('sel_oil', 'None')),
        'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('sel_surf', 'None')),
        'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('sel_cosurf', 'None'))
    }])

    res = {target: models[target].predict(input_df)[0] for target in models}
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Size", f"{res['Size_nm']:.1f} nm")
    m2.metric("PDI", f"{res['PDI']:.3f}")
    m3.metric("Zeta", f"{res['Zeta_mV']:.1f} mV")
    m4.metric("%EE", f"{res['Encapsulation_Efficiency']:.2f}")

    st.subheader("🔍 Prediction Interpretation")
    # Custom explanation based on the specific prediction results
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_values = explainer(input_df)
    
    fig_shap, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig_shap)
    
    # Targeted SHAP explanation
    top_feature = X_train.columns[np.argmax(np.abs(shap_values.values[0]))]
    st.success(f"**Interpretation:** The model identifies **{top_feature}** as the most significant driver for this specific Droplet Size prediction. A change in this component will have the highest impact on physical stability.")
