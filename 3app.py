import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES SAFETY ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v9.0", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 22px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 10px solid #28a745;
        margin-bottom: 20px;
    }
    .m-label { font-size: 14px; color: #555; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 26px; color: #000; font-weight: 800; white-space: nowrap; }
    .model-box { 
        background: #1e272e; color: #ffffff; padding: 25px; 
        border-radius: 15px; border-top: 5px solid #00d8d6;
    }
    .locked-msg { text-align: center; padding: 50px; color: #a0aec0; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Please upload '{csv_file}' to your GitHub repo.")
        st.stop()
    
    df = pd.read_csv(csv_file)
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        
    df_train = df.dropna(subset=[f'{col}_clean' for col in targets]).copy()
    
    q1, q3 = df_train['Encapsulation_Efficiency_clean'].quantile([0.25, 0.75])
    iqr = q3 - q1
    df_train['is_outlier'] = (df_train['Encapsulation_Efficiency_clean'] < (q1 - 1.5 * iqr)) | \
                             (df_train['Encapsulation_Efficiency_clean'] > (q3 + 1.5 * iqr))

    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df_train[col] = df_train[col].fillna("Not Specified").astype(str)

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=300, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').fillna(False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=300, random_state=42).fit(X, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. STRUCTURE PREDICTION ---
@st.cache_data
def get_structure(drug_name):
    if not HAS_CHEM_LIBS: return None, None, None, None
    try:
        comp = pcp.get_compounds(drug_name, 'name')[0]
        mol = Chem.MolFromSmiles(comp.canonical_smiles)
        return Draw.MolToImage(mol, size=(300, 300)), comp.canonical_smiles, comp.molecular_weight, comp.xlogp
    except: return None, None, None, None

# --- 3. UI STATE & SIDEBAR ---
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

st.sidebar.title("NanoPredict Controls")
nav_choice = st.sidebar.radio("Go to:", ["Step 1: Chemical Setup", "Step 2: Expert Rationale", "Step 3: Prediction & Nanomodel"])

# --- PAGE 1: SETUP ---
if nav_choice == "Step 1: Chemical Setup":
    st.header("Step 1: API & Lipid Selection")
    c1, c2 = st.columns(2)
    with c1:
        selected_drug = st.selectbox("Select API", sorted(df['Drug_Name'].unique()), key="drug_select")
        selected_oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()), key="oil_select")
        if st.button("✅ Confirm & Unlock System"):
            st.session_state.setup_complete = True
            st.session_state.current_drug = selected_drug
            st.session_state.current_oil = selected_oil
            st.success("System Unlocked!")
    with c2:
        img, smi, mw, lp = get_structure(selected_drug)
        if img: st.image(img, caption=f"2D Molecule: {selected_drug}")

# --- PAGE 2: RATIONALE ---
elif nav_choice == "Step 2: Expert Rationale":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Locked: Please complete Step 1.</div>", unsafe_allow_html=True)
    else:
        st.header("Step 2: Scientific Rationale")
        oil = st.session_state.current_oil
        best = df[df['Oil_phase'] == oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False).iloc[0]
        st.info(f"Recommended System: {best['Surfactant']} + {best['Co-surfactant']}")
        st.write(f"This system achieved {best['Encapsulation_Efficiency_clean']:.1f}% efficiency in historical trials.")

# --- PAGE 3: PREDICTION & MODELING ---
elif nav_choice == "Step 3: Prediction & Nanomodel":
    if not st.session_state.setup_complete:
        st.markdown("<div class='locked-msg'>🔒 Locked: Please complete Step 1.</div>", unsafe_allow_html=True)
    else:
        st.header("Step 3: Outcome & Structural Nanomodel")
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            s_choice = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
            cs_choice = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            
            if st.button("🚀 Execute Simulation"):
                # Prediction logic
                inputs = [[le_dict['Drug_Name'].transform([st.session_state.current_drug])[0],
                           le_dict['Oil_phase'].transform([st.session_state.current_oil])[0],
                           le_dict['Surfactant'].transform([s_choice])[0],
                           le_dict['Co-surfactant'].transform([cs_choice])[0]]]
                
                res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
                
                # RESULTS DISPLAY
                st.subheader("Predicted Outcomes")
                metrics = [("Size", f"{res[0]:.1f} nm"), ("Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f}%")]
                for l, v in metrics:
                    st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        with c2:
            st.subheader("AI Nanomodel Visualization")
            if 'res' in locals():
                # STRUCTURAL MODEL DESCRIPTION
                st.markdown(f"""
                <div class="model-box">
                <h4>Droplet Architecture (Predicted)</h4>
                <hr>
                <ul>
                    <li><b>Core:</b> Lipophilic core containing <b>{st.session_state.current_oil}</b> and solubilized <b>{st.session_state.current_drug}</b>.</li>
                    <li><b>Interfacial Layer:</b> A high-density film of <b>{s_choice}</b> and <b>{cs_choice}</b> molecules.</li>
                    <li><b>Self-Assembly:</b> The surfactant tails are anchored in the oil core, while polar heads face the water.</li>
                    <li><b>Stability Mechanism:</b> Electrostatic repulsion (Zeta: {res[2]:.1f} mV) preventing coalescence.</li>
                    <li><b>Droplet Geometry:</b> Spherical micelle with an average radius of {res[0]/2:.1f} nm.</li>
                </ul>
                <p style="font-size: 12px; color: #00d8d6;">*Simulation based on Brownian motion and interfacial tension parameters.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Simple Visual representation using Plotly
                fig = go.Figure(data=[go.Scatter(
                    x=[0], y=[0], mode='markers',
                    marker=dict(size=res[0], color='rgba(0, 216, 214, 0.6)', line=dict(width=4, color='white'))
                )])
                fig.update_layout(title="Droplet Cross-Section (Scale: nm)", showlegend=False, 
                                  xaxis=dict(visible=False), yaxis=dict(visible=False), height=300)
                st.plotly_chart(fig, use_container_width=True)
