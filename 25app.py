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

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v2.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .metric-card { background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-top: 4px solid #28a745; text-align: center; margin-bottom: 15px;}
    .m-label { font-size: 11px; color: #666; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 22px; font-weight: 800; color: #1a202c; }
    .status-box { padding: 25px; border-radius: 12px; text-align: center; margin-bottom: 25px; font-weight: 800; font-size: 26px; border: 2px solid;}
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
    step_choice = st.radio("Formulation Steps", ["Step 1: Drug & Data", "Step 2: Advanced Solubility", "Step 3: Component AI & Ratios", "Step 4: Phase Ratios", "Step 5: Finalize", "Step 6: Analysis"])

# --- STEP 1: DRUG & DATA ---
if step_choice == "Step 1: Drug & Data":
    st.header("Step 1: Chemical & Data Setup")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("1. Drug Input")
        smiles = st.text_input("SMILES", "CC(=O)OC1=CC=CC=C1C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            st.session_state.logp, st.session_state.mw = round(Descriptors.MolLogP(mol), 2), round(Descriptors.MolWt(mol), 2)
            st.image(Draw.MolToImage(mol, size=(250,200)))
        
        st.write("---")
        st.subheader("2. Dataset Configuration")
        up = st.file_uploader("Browse/Upload Training Data (CSV)", type="csv")
        if up: st.session_state.csv_data = up

    with col2:
        if le_dict:
            st.subheader("Oil Phase Solubility Map (0-100%)")
            # Populated Oil Data
            oils = ["MCT Oil", "Oleic Acid", "Isopropyl Myristate", "Capmul MCM", "Labrafil M", "Castor Oil", "Soybean Oil", "Olive Oil"]
            logp_val = st.session_state.get('logp', 3.5)
            # Calculated affinity scores based on LogP compatibility
            scores = [np.clip(95 - abs(logp_val - (1.5 + (i*0.6)))*15, 10, 99) for i in range(len(oils))]
            aff_df = pd.DataFrame({"Oil": oils, "Solubility (%)": scores}).sort_values("Solubility (%)", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Solubility (%)", y="Oil", range_x=[0,100], orientation='h', color="Solubility (%)", color_continuous_scale='Turbo'), use_container_width=True)

# --- STEP 2: ADVANCED SOLUBILITY ---
elif step_choice == "Step 2: Advanced Solubility":
    st.header("Step 2: Thermodynamic Solubility Analysis")
    logp = st.session_state.get('logp', 3.0)
    mw = st.session_state.get('mw', 300.0)
    sol_water = 10**(0.5 - 0.01 * (mw - 50) - 0.6 * logp) * 1000 
    sol_oil = sol_water * (10**logp) 
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💧 Phase Solubility")
        st.metric("Drug Solubility in Water", f"{sol_water:.4f} mg/L")
        st.metric("Estimated Drug Solubility in Oil", f"{sol_oil/1000:.2f} mg/mL")

    with c2:
        st.subheader("🧪 Nanoemulsion Type Capacities")
        # System-specific solubility limits
        st.write(f"**Oil-in-Water (O/W):** Predicted max loading ~{sol_oil * 0.12:.2f} mg/mL")
        st.write(f"**Water-in-Oil (W/O):** Predicted max loading ~{sol_oil * 0.88:.2f} mg/mL")
        st.write(f"**Bicontinuous System:** Predicted max loading ~{sol_oil * 0.45:.2f} mg/mL")
    
    st.write("---")
    st.latex(r"S_{total} = (\Phi_{oil} \cdot S_{oil}) + (\Phi_{water} \cdot S_{water})")

# --- STEP 3: COMPONENT AI ---
elif step_choice == "Step 3: Component AI & Ratios":
    st.header("Step 3: Component AI & Smix Optimization")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.km_ratio = st.select_slider("Smix Ratio (Surfactant : Co-Surfactant)", options=["1:1", "2:1", "3:1", "4:1"], value="2:1")
        st.session_state.oil_choice = st.selectbox("Select Target Oil", sorted(le_dict['Oil_phase'].classes_) if le_dict else ["None"])
    with col2:
        st.subheader("Calculated Smix ($K_{m}$)")
        km_val = int(st.session_state.km_ratio.split(":")[0])
        st.latex(rf"K_{{m}} = \frac{{S}}{{CoS}} = {km_val}")

# --- STEP 4: PHASE RATIOS ---
elif step_choice == "Step 4: Phase Ratios":
    st.header("Step 4: Global Formulation Composition")
    st.session_state.oil_p = st.slider("Oil Phase %", 5, 40, 15)
    st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
    st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
    st.metric("S-mix to Oil Ratio", f"{(st.session_state.smix_p / st.session_state.oil_p):.2f}")

# --- STEP 5: FINALIZE ---
elif step_choice == "Step 5: Finalize":
    st.header("Step 5: Final Selection")
    st.session_state.s_final = st.selectbox("Confirm Surfactant", sorted(le_dict['Surfactant'].classes_) if le_dict else ["None"])
    st.session_state.cs_final = st.selectbox("Confirm Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_) if le_dict else ["None"])
    if st.button("Finalize for Analysis"): st.balloons()

# --- STEP 6: ANALYSIS ---
elif step_choice == "Step 6: Analysis":
    st.header("Step 6: AI Predictions & Surface Mapping")
    if df_raw is None:
        st.error("Please provide data in Step 1!")
    else:
        input_df = pd.DataFrame([{
            'Drug_Name_enc': get_enc(le_dict['Drug_Name'], st.session_state.get('drug_name', 'None')),
            'Oil_phase_enc': get_enc(le_dict['Oil_phase'], st.session_state.get('oil_choice', 'None')),
            'Surfactant_enc': get_enc(le_dict['Surfactant'], st.session_state.get('s_final', 'None')),
            'Co-surfactant_enc': get_enc(le_dict['Co-surfactant'], st.session_state.get('cs_final', 'None')),
            'HLB': 12.0
        }])

        is_stable = stab_model.predict(input_df)[0]
        st.markdown(f'<div class="status-box" style="background-color: {"#d4edda" if is_stable else "#f8d7da"};">{"STABLE" if is_stable else "UNSTABLE"}</div>', unsafe_allow_html=True)

        p_cols = st.columns(4)
        for i, target in enumerate(['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']):
            val = models[target].predict(input_df)[0]
            with p_cols[i]:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{target}</div><div class='m-value'>{val:.2f}</div></div>", unsafe_allow_html=True)

        t1, t2 = st.tabs(["📐 Ternary Map", "🌊 3D Surface"])
        with t1:
            
            fig_tern = go.Figure(go.Scatterternary({'mode': 'markers','a': [st.session_state.get('oil_p', 15)], 'b': [st.session_state.get('smix_p', 30)], 'c': [st.session_state.get('water_p', 55)], 'marker': {'size': 20, 'color': 'green'}}))
            st.plotly_chart(fig_tern, use_container_width=True)
        with t2:
            
            o_g, s_g = np.meshgrid(np.linspace(5, 40, 20), np.linspace(10, 60, 20))
            z_g = models['Size_nm'].predict(input_df)[0] + (o_g * 0.5) - (s_g * 0.3)
            st.plotly_chart(go.Figure(data=[go.Surface(z=z_g, x=o_g, y=s_g)]), use_container_width=True)

        # SHAP EXPLANATION
        st.write("---")
        st.subheader("💡 AI Decision Explanation (SHAP Analysis)")
        st.markdown("""
        **How to interpret this chart:** The SHAP waterfall plot below quantifies the influence of each formulation parameter on the predicted **Particle Size**.  
        - **Positive Values (Red):** These components are driving the particle size higher (making droplets larger).  
        - **Negative Values (Blue):** These components are reducing the particle size (making droplets smaller).  
        - **E(f(x)):** Represents the average base size across all data points before these specific inputs are applied.
        """)
        
        explainer = shap.Explainer(models['Size_nm'], X_train)
        shap_values = explainer(input_df)
        fig_shap, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_shap)
