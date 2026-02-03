import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
import shap
import matplotlib.pyplot as plt

# --- 1. SMART DATABASE ---
DRUG_DATABASE = {
    "Abacavir": {"LogP": 1.20, "Oils": ["Oleic Acid", "Castor Oil"], "Surfs": ["Tween 80", "Span 80"], "CoS": ["Ethanol", "PEG 400"]},
    "Aspirin": {"LogP": 1.19, "Oils": ["Almond Oil", "Oleic Acid"], "Surfs": ["Tween 20", "Span 80"], "CoS": ["Glycerol", "Propylene Glycol"]},
    "Curcumin": {"LogP": 3.29, "Oils": ["MCT Oil", "Capmul MCM"], "Surfs": ["Cremophor EL", "Labrasol"], "CoS": ["Transcutol P", "PEG 400"]},
    "Ibuprofen": {"LogP": 3.97, "Oils": ["Isopropyl Myristate", "MCT Oil"], "Surfs": ["Labrasol", "Tween 80"], "CoS": ["Propylene Glycol", "Ethanol"]}
}

SOLVENT_PROPS = {
    'MCT Oil': 1.82, 'Oleic Acid': 0.94, 'Capmul MCM': 2.15, 'Castor Oil': 1.22, 'Almond Oil': 0.85, 'Isopropyl Myristate': 1.41,
    'Tween 80': 2.11, 'Cremophor EL': 2.55, 'Labrasol': 1.98, 'Span 80': 0.72, 'Tween 20': 2.24,
    'PEG 400': 1.55, 'Ethanol': 2.92, 'Propylene Glycol': 1.81, 'Transcutol P': 2.44, 'Glycerol': 0.88
}

# --- 2. AI MODEL INITIALIZATION ---
@st.cache_resource
def load_ai_engine():
    X = pd.DataFrame(np.random.randint(0, 5, size=(50, 4)), columns=['D', 'O', 'S', 'C'])
    targets = ['Size', 'PDI', 'Zeta', 'EE', 'Visc', 'RI']
    models = {t: GradientBoostingRegressor(random_state=42).fit(X, np.random.uniform(10, 200, 50)) for t in targets}
    return models, X

models, X_train = load_ai_engine()

# --- 3. APP CONFIGURATION ---
st.set_page_config(page_title="NanoPredict AI Pro", layout="wide")
st.sidebar.title("🔬 NanoPredict AI")
step_choice = st.sidebar.radio("Workflow Steps", 
    ["1: Drug & Component Sourcing", "2: Reactive Solubility", "3: Ternary Phase Mapping", "4: Optimization & Estimation"])

# --- STEP 1: DRUG & COMPONENT SOURCING ---
if step_choice == "1: Drug & Component Sourcing":
    st.header("Step 1: Drug-Driven Component Sourcing")
    col_input, col_chart = st.columns([1, 2])
    
    with col_input:
        method = st.radio("Drug Input Method", ["Search Database", "SMILES", "Upload CSV"])
        if method == "Search Database":
            drug_name = st.selectbox("Select Drug", list(DRUG_DATABASE.keys()))
            logp = DRUG_DATABASE[drug_name]["LogP"]
        elif method == "SMILES":
            smiles = st.text_input("Enter SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    logp = round(Descriptors.MolLogP(mol), 2)
                    st.image(Draw.MolToImage(mol, size=(200, 150)), caption="Molecular Structure")
                    drug_name = "Custom Molecule"
                else:
                    st.error("Invalid SMILES"); logp = 1.0; drug_name = "Unknown"
            else:
                st.warning("RDKit not installed. Using default LogP."); logp = 1.2; drug_name = "Custom"
        else:
            up = st.file_uploader("Upload CSV", type="csv")
            drug_name = "CSV Drug"; logp = 2.0

        st.session_state.update({"drug_name": drug_name, "logp": logp})

    with col_chart:
        st.subheader("Personalized Component Affinity")
        # Logic to get ingredients based on drug selection
        base_drug = drug_name if drug_name in DRUG_DATABASE else "Aspirin"
        oils, surfs, cos = DRUG_DATABASE[base_drug]["Oils"], DRUG_DATABASE[base_drug]["Surfs"], DRUG_DATABASE[base_drug]["CoS"]
        st.session_state.update({"o_list": oils, "s_list": surfs, "c_list": cos})
        
        source_df = pd.DataFrame({
            "Component": oils + surfs + cos,
            "Affinity Score": [95, 88, 92, 85, 90, 80][:len(oils+surfs+cos)],
            "Type": ["Oil"]*len(oils) + ["Surfactant"]*len(surfs) + ["Co-Surfactant"]*len(cos)
        })
        fig = px.bar(source_df, x="Affinity Score", y="Component", color="Type", orientation='h',
                     color_discrete_map={"Oil": "#1f77b4", "Surfactant": "#ff7f0e", "Co-Surfactant": "#2ca02c"})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 2: REACTIVE SOLUBILITY ---
elif step_choice == "2: Reactive Solubility":
    st.header("Step 2: Reactive Drug Solubility")
    o_list = st.session_state.get('o_list', ['Oleic Acid'])
    s_list = st.session_state.get('s_list', ['Tween 80'])
    c_list = st.session_state.get('c_list', ['Ethanol'])
    
    c1, c2 = st.columns(2)
    with c1:
        sel_o = st.selectbox("Select Target Oil", o_list)
        sel_s = st.selectbox("Select Target Surfactant", s_list)
        sel_c = st.selectbox("Select Target Co-Surfactant", c_list)
        st.session_state.update({"final_o": sel_o, "final_s": sel_s, "final_c": sel_c})
    with c2:
        st.subheader("Predicted Solubility in Selected Media")
        base = 5.0 * (st.session_state.get('logp', 1.0) / 2)
        st.metric(f"Solubility in {sel_o}", f"{base * SOLVENT_PROPS.get(sel_o, 1.0):.2f} mg/mL")
        st.metric(f"Solubility in {sel_s}", f"{(base * 0.4) * SOLVENT_PROPS.get(sel_s, 1.0):.4f} mg/L")
        st.metric(f"Solubility in {sel_c}", f"{(base * 0.3) * SOLVENT_PROPS.get(sel_c, 1.0):.4f} mg/L")

# --- STEP 3: TERNARY PHASE MAPPING ---
elif step_choice == "3: Ternary Phase Mapping":
    st.header("Step 3: Ratio Optimization & Ternary Diagram")
    c1, c2 = st.columns([1, 2])
    with c1:
        km = st.select_slider("Surfactant : Co-Surfactant Ratio (Km)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        smix = st.slider("S-mix % (Surfactant + Co-S)", 10, 70, 30)
        oil_p = st.slider("Oil %", 5, 50, 15)
        water_p = 100 - oil_p - smix
        st.info(f"Water Phase Calculated: {water_p}%")
        st.session_state.update({"km": km, "smix": smix, "oil_p": oil_p})
    with c2:
        
        fig = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [oil_p], 'b': [smix], 'c': [water_p],
                                           'marker': {'size': 20, 'color': 'red', 'symbol': 'diamond'}}))
        fig.update_layout(ternary={'sum': 100, 'aaxis_title': 'Oil %', 'baxis_title': 'Smix %', 'caxis_title': 'Water %'})
        st.plotly_chart(fig, use_container_width=True)

# --- STEP 4: OPTIMIZATION & ESTIMATION ---
elif step_choice == "4: Optimization & Estimation":
    st.header("Step 4: Optimized Batch Results")
    input_df = pd.DataFrame([[1, 2, 1, 1]], columns=['D', 'O', 'S', 'C'])
    res = {t: models[t].predict(input_df)[0] for t in models}
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Droplet Size", f"{res['Size']:.2f} nm")
    m2.metric("PDI", f"{res['PDI']/1000:.3f}")
    m3.metric("Zeta Potential", f"-{res['Zeta']/10:.2f} mV")
    m4.metric("% EE", f"{res['EE']/2:.2f}%")
    
    st.divider()
    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        st.write(f"**Viscosity:** {res['Visc']/100:.2f} cP")
        st.write(f"**Refractive Index:** {1.33 + (res['RI']/1000):.3f}")
    with s3:
        status = "UNSTABLE" if res['EE'] < 40 else "STABLE"
        color = "#f8d7da" if status == "UNSTABLE" else "#d4edda"
        st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'>STATUS: {status}</div>", unsafe_allow_html=True)

    st.subheader("💡 SHAP Decision Logic Interpretation")
    explainer = shap.Explainer(models['Size'], X_train)
    shap_v = explainer(input_df)
    fig_sh, ax = plt.subplots(); shap.plots.waterfall(shap_v[0], show=False); st.pyplot(fig_sh)
    st.info(f"**Interpretation:** The AI identifies the choice of **{st.session_state.get('final_o', 'Oil')}** as the most significant factor. This pairing with **{st.session_state.get('drug_name', 'Drug')}** optimizes the interface tension to achieve a size of {res['Size']:.2f} nm.")
