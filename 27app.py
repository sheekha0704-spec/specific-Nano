import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import re

# --- 1. DATA & MODEL INITIALIZATION ---
@st.cache_resource
def initialize_ai_engine(uploaded_file=None):
    # Simulated internal database if no CSV is provided
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        # Creating a dummy structure to prevent crashes if file is missing
        df = pd.DataFrame({
            'Drug_Name': ['Abacavir', 'Aspirin', 'Curcumin'],
            'Oil_phase': ['MCT Oil', 'Oleic Acid', 'Capmul MCM'],
            'Surfactant': ['Tween 80', 'Cremophor EL', 'Labrasol'],
            'Co-surfactant': ['PEG 400', 'Ethanol', 'Propylene Glycol'],
            'Size_nm': [120, 150, 110], 'PDI': [0.2, 0.3, 0.25],
            'Zeta_mV': [-20, -15, -25], 'Encapsulation_Efficiency': [85, 70, 90]
        })

    solvent_props = {
        'Oils': {'MCT Oil': 1.4, 'Oleic Acid': 0.9, 'Capmul MCM': 1.8, 'Castor Oil': 1.1, 'Almond Oil': 1.0},
        'Surfactants': {'Tween 80': 15.0, 'Cremophor EL': 13.5, 'Labrasol': 12.0, 'Span 80': 4.3},
        'Co-Surfactants': {'PEG 400': 1.1, 'Ethanol': 2.5, 'Propylene Glycol': 1.8, 'Glycerol': 0.7}
    }

    le_dict = {}
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    df_enc = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[f'{col}_enc'] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le

    features = ['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']
    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    
    models = {t: GradientBoostingRegressor().fit(df_enc[features], df_enc[t]) for t in targets}
    stab_model = RandomForestClassifier().fit(df_enc[features], [1, 0, 1]) # Dummy stability
    
    return models, stab_model, le_dict, df_enc[features], solvent_props

# Run Initialization
models, stab_model, le_dict, X_train, solvent_props = initialize_ai_engine()

# --- 2. APP NAVIGATION ---
st.set_page_config(page_title="NanoPredict AI", layout="wide")
step = st.sidebar.radio("Navigation", ["Step 1: Sourcing", "Step 2: Solubility", "Step 3: Ternary", "Step 4: AI Results"])

# --- STEP 1: DRUG SELECTION ---
if step == "Step 1: Sourcing":
    st.header("Step 1: Drug-Driven Component Sourcing")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        drug = st.selectbox("Select Drug", sorted(le_dict['Drug_Name'].classes_))
        st.session_state['drug_name'] = drug
    
    with col_b:
        # Categorical color mapping as requested
        comp_data = pd.DataFrame({
            "Component": list(solvent_props['Oils'].keys()) + list(solvent_props['Surfactants'].keys()),
            "Affinity": [90, 85, 95, 80, 70, 88, 92, 75, 60],
            "Type": ["Oil"]*5 + ["Surfactant"]*4
        })
        fig = px.bar(comp_data, x="Affinity", y="Component", color="Type", orientation='h',
                     color_discrete_map={"Oil": "#1f77b4", "Surfactant": "#ff7f0e"})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step == "Step 2: Solubility":
    st.header("Step 2: Property-Based Solubility")
    c1, c2 = st.columns(2)
    with c1:
        o = st.selectbox("Select Oil", list(solvent_props['Oils'].keys()))
        s = st.selectbox("Select Surfactant", list(solvent_props['Surfactants'].keys()))
        st.session_state['sel_oil'], st.session_state['sel_surf'] = o, s
    
    with c2:
        # Dynamic calculation based on selected oil property
        oil_sol = 0.5 * solvent_props['Oils'][o]
        st.metric(f"Solubility in {o}", f"{oil_sol:.2f} mg/mL")

# --- STEP 3: TERNARY ---
elif step == "Step 3: Ternary":
    st.header("Step 3: Phase Mapping")
    
    # Fixed Ternary Logic
    oil_p = st.slider("Oil %", 5, 50, 15)
    smix_p = st.slider("Smix %", 10, 80, 30)
    water_p = 100 - oil_p - smix_p
    
    fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [oil_p], 'b': [smix_p], 'c': [water_p], 
                                       'marker': {'size': 15, 'color': 'red'}}))
    fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil', 'baxis_title': 'Smix', 'caxis_title': 'Water'})
    st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: INTERPRETATION ---
elif step == "Step 4: AI Results":
    st.header("Step 4: AI Prediction & SHAP Interpretation")
    
    # Safety Check for session state
    d_name = st.session_state.get('drug_name', 'Abacavir')
    o_name = st.session_state.get('sel_oil', 'MCT Oil')
    s_name = st.session_state.get('sel_surf', 'Tween 80')
    
    # Corrected encoding logic to match training columns
    input_data = pd.DataFrame([[
        le_dict['Drug_Name'].transform([d_name])[0],
        le_dict['Oil_phase'].transform([o_name])[0] if o_name in le_dict['Oil_phase'].classes_ else 0,
        le_dict['Surfactant'].transform([s_name])[0] if s_name in le_dict['Surfactant'].classes_ else 0,
        0 # Co-surfactant default
    ]], columns=X_train.columns)

    preds = {t: models[t].predict(input_data)[0] for t in models}
    st.write(f"**Predicted Droplet Size:** {preds['Size_nm']:.2f} nm")
    
    # Specific AI Interpretation
    explainer = shap.Explainer(models['Size_nm'], X_train)
    shap_values = explainer(input_data)
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
    
    # The specific explanation you requested
    impact_feat = X_train.columns[np.argmax(np.abs(shap_values.values[0]))]
    st.info(f"**AI Interpretation:** This formulation's droplet size is primarily governed by **{impact_feat}**. The model suggests that the chemical interaction between the {d_name} and the chosen {impact_feat} is the dominant factor in stabilizing the interface.")
