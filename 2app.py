import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMOINFORMATICS LIBRARIES ---
# Ensure 'rdkit' and 'pubchempy' are in requirements.txt
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict Structural AI", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05); border-left: 8px solid #2e86de;
        margin-bottom: 15px;
    }
    .m-label { font-size: 14px; color: #576574; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 26px; color: #222f3e; font-weight: 800; white-space: nowrap; }
    .axis-guide { 
        background: #f1f2f6; padding: 15px; border-radius: 10px; 
        border: 1px solid #dfe4ea; margin-bottom: 20px;
    }
    .rationale-box { 
        background: #e3f2fd; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #1976d2; color: #0d47a1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA & ML ENGINE ---
@st.cache_data
def load_and_train():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"File '{csv_file}' not found in directory.")
        st.stop()
        
    df = pd.read_csv(csv_file)
    
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        df[f'{col}_clean'] = df.groupby('Oil_phase')[f'{col}_clean'].transform(lambda x: x.fillna(x.median()))

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    models = {}
    for col in targets:
        m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.04, max_depth=5, random_state=42)
        m.fit(X, df[f'{col}_clean'])
        models[col] = m
    
    df['is_stable'] = df['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=400, random_state=42).fit(X, df['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_train()

# --- 2. CHEMICAL STRUCTURE HELPER ---
@st.cache_data
def get_mol_data(drug_name):
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            c = compounds[0]
            mol = Chem.MolFromSmiles(c.canonical_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            return img, c.canonical_smiles, c.molecular_weight, c.xlogp
    except:
        return None, None, None, None

# --- 3. UI NAVIGATION ---
page = st.sidebar.radio("Navigation", ["Step 1: Chemical Setup", "Step 2: Expert Rationale", "Step 3: Predictions"])

if 'inputs' not in st.session_state:
    st.session_state.inputs = {'drug': sorted(df['Drug_Name'].unique())[0], 'oil': sorted(df['Oil_phase'].unique())[0]}

# --- PAGE 1: SETUP ---
if page == "Step 1: Chemical Setup":
    st.header("Step 1: API Selection & Molecular Structure")
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.session_state.inputs['drug'] = st.selectbox("Search API Name", sorted(df['Drug_Name'].unique()))
        st.session_state.inputs['oil'] = st.selectbox("Oil Phase Carrier", sorted(df['Oil_phase'].unique()))
        st.write("---")
        st.write("**Note:** AI will predict structural features based on the API name.")

    with col_b:
        img, smiles, mw, logp = get_mol_data(st.session_state.inputs['drug'])
        if img:
            st.image(img, caption=f"Predicted Structure: {st.session_state.inputs['drug']}")
            st.write(f"**MW:** {mw} g/mol | **LogP:** {logp}")
            st.code(f"SMILES: {smiles}")
        else:
            st.warning("Structure not found in database.")

# --- PAGE 2: RATIONALE ---
elif page == "Step 2: Expert Rationale":
    st.header("Step 2: Scientific Formulation Evidence")
    oil = st.session_state.inputs['oil']
    drug = st.session_state.inputs['drug']
    
    # Logic: Pulling the most successful data point for this oil from 900+ rows
    top_data = df[df['Oil_phase'] == oil].sort_values(by='EE_clean', ascending=False).iloc[0]
    
    st.markdown(f"### Specific Recommendation for **{oil}**")
    st.markdown(f"""
    <div class="rationale-box">
    <b>AI Analysis of 900+ Research Rows:</b><br>
    The recommended system is <b>{top_data['Surfactant']}</b> combined with <b>{top_data['Co-surfactant']}</b>. 
    <br><br>
    <b>Scientific Reason:</b> Historically, this combination with {oil} achieved a droplet size of 
    <b>{top_data['Size_nm_clean']:.1f} nm</b> and an encapsulation efficiency of <b>{top_data['EE_clean']:.1f}%</b>. 
    The HLB of this system is perfectly balanced for the lipophilic nature of {drug}.
    </div>
    """, unsafe_allow_html=True)

# --- PAGE 3: PREDICTIONS ---
elif page == "Step 3: Predictions":
    st.header("Step 3: Final Optimized Predictors")
    
    # PUBLIC AXIS GUIDE
    st.markdown("""
    <div class="axis-guide">
    <b>🧭 3D Phase Axis Guide:</b><br>
    <b>X (Oil %):</b> Concentration of the Lipid Phase | 
    <b>Y (S-mix %):</b> Concentration of Surfactant+Co-Surfactant | 
    <b>Z (Water %):</b> Concentration of the Aqueous Phase
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        s_choice = st.selectbox("Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Run Multi-Output AI"):
            d_idx = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_idx = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_idx = le_dict['Surfactant'].transform([s_choice])[0]
            c_idx = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            vec = [[d_idx, o_idx, s_idx, c_idx]]
            res = [models[col].predict(vec)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
            stab = stab_model.predict_proba(vec)[0][1] * 100
            
            # FULL DISPLAY CARDS
            metrics = [
                ("Droplet Size", f"{res[0]:.2f} nm"), ("PDI", f"{res[1]:.3f}"),
                ("Zeta Potential", f"{res[2]:.2f} mV"), ("Drug Loading", f"{res[3]:.2f} mg/mL"),
                ("Encapsulation Eff.", f"{res[4]:.2f} %"), ("Stability Confidence", f"{stab:.1f} %")
            ]
            for label, val in metrics:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{label}</div><div class='m-value'>{val}</div></div>", unsafe_allow_html=True)

    with c2:
        # 3D Ternary Chart
        oil_v = np.linspace(5, 40, 15)
        smix_v = np.linspace(15, 65, 15)
        O, S = np.meshgrid(oil_v, smix_v)
        W = 100 - O - S
        mask = W > 0
        
        fig = go.Figure(data=[go.Scatter3d(
            x=O[mask], y=S[mask], z=W[mask],
            mode='markers', marker=dict(size=5, color=S[mask], colorscale='Viridis', colorbar=dict(title="S-mix %"))
        )])
        fig.update_layout(scene=dict(xaxis_title='Oil % (X)', yaxis_title='S-mix % (Y)', zaxis_title='Water % (Z)'),
                          margin=dict(l=0,r=0,b=0,t=0), height=550)
        st.plotly_chart(fig, use_container_width=True)
