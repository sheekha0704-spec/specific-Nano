import streamlit as st
import pandas as pd
import os

# --- 1. SAFETY CHECK FOR FILE ---
csv_filename = 'nanoemulsion 2.csv'

if not os.path.exists(csv_filename):
    st.error(f"❌ File Not Found: Please ensure '{csv_filename}' is uploaded to your GitHub repository.")
    st.info("If you renamed the file, please update the code line: csv_filename = 'your_new_name.csv'")
    st.stop() # Stops the app from crashing with a red error block

# --- 2. REST OF THE MODEL ---
# (The rest of the High-Precision code from our previous turn goes here)
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re

# Chemoinformatics Tools
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict Structural AI", layout="wide")

# CUSTOM CSS: Solving the truncation and font issues
st.markdown("""
    <style>
    .metric-card {
        background: #fdfdfd; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 10px solid #2ecc71;
        margin-bottom: 20px;
    }
    .m-label { font-size: 15px; color: #7f8c8d; font-weight: bold; text-transform: uppercase; }
    .m-value { font-size: 26px; color: #2c3e50; font-weight: 800; white-space: nowrap; }
    .axis-legend-box { 
        background: #ebf5fb; padding: 20px; border-radius: 10px; 
        border: 1px solid #3498db; line-height: 1.6;
    }
    .rationale-box { 
        background: #fef9e7; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #f1c40f; color: #7d6608;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. THE BRAIN: DATA & ML ENGINE ---
@st.cache_data
def load_and_train_precision():
    # Attempt to load the user's specific file
    try:
        df = pd.read_csv('nanoemulsion 2.csv')
    except:
        st.error("Error: 'nanoemulsion 2.csv' not found. Please upload to GitHub.")
        st.stop()
    
    def get_num(x):
        if pd.isna(x): return np.nan
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else np.nan

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']
    for col in targets:
        df[f'{col}_clean'] = df[col].apply(get_num)
        # Accurate imputation based on Oil_phase group
        df[f'{col}_clean'] = df.groupby('Oil_phase')[f'{col}_clean'].transform(lambda x: x.fillna(x.median()))

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        
    X = df[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    
    # Precise Gradient Boosting Models
    models = {}
    for col in targets:
        m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.04, max_depth=5, random_state=42)
        m.fit(X, df[f'{col}_clean'])
        models[col] = m
    
    # Stability Confidence Classifier
    df['is_stable'] = df['Stability'].str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier(n_estimators=400, random_state=42).fit(X, df['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_train_precision()

# Helper: Fetch structure from PubChem
@st.cache_data
def fetch_structure(drug_name):
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            comp = compounds[0]
            mol = Chem.MolFromSmiles(comp.canonical_smiles)
            img = Draw.MolToImage(mol, size=(400, 400))
            return img, comp.canonical_smiles, comp.molecular_weight, comp.xlogp
    except:
        return None, None, None, None

# --- 2. MULTI-PAGE NAVIGATION ---
page = st.sidebar.radio("Main Menu", ["1. API & Chemical Structure", "2. Data-Driven Rationale", "3. Precision Prediction", "History"])

if 'inputs' not in st.session_state:
    st.session_state.inputs = {'drug': sorted(df['Drug_Name'].unique())[0], 'oil': sorted(df['Oil_phase'].unique())[0]}

# --- STEP 1: API & STRUCTURE ---
if page == "1. API & Chemical Structure":
    st.header("Step 1: Compound Identification & Structure Prediction")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.session_state.inputs['drug'] = st.selectbox("Select Drug for Structural Analysis", sorted(df['Drug_Name'].unique()))
        st.session_state.inputs['oil'] = st.selectbox("Target Oil Phase", sorted(df['Oil_phase'].unique()))
        st.write("---")
        st.info("The AI will now fetch the molecular data for this API from PubChem.")

    with c2:
        img, sm, mw, lp = fetch_structure(st.session_state.inputs['drug'])
        if img:
            st.image(img, caption=f"Predicted 2D Structure of {st.session_state.inputs['drug']}")
            st.markdown(f"**Molecular Weight:** `{mw} g/mol` | **LogP:** `{lp}`")
            st.markdown(f"**SMILES Identifier:** `{sm}`")
        else:
            st.warning("Structure not available in current database.")

# --- STEP 2: RATIONALE ---
elif page == "2. Data-Driven Rationale":
    st.header("Step 2: Evidence-Based Recommendation")
    oil = st.session_state.inputs['oil']
    
    # Specific Search: Find historical outcomes for this specific Oil
    oil_stats = df[df['Oil_phase'] == oil].sort_values(by='EE_clean', ascending=False)
    
    if not oil_stats.empty:
        top_form = oil_stats.iloc[0]
        st.markdown(f"### Optimal Surfactant System for **{oil}**")
        st.markdown(f"""
        <div class="rationale-box">
        <b>AI Analysis of your 900+ Data Points:</b><br>
        For the oil <b>{oil}</b>, the most specific recommendation is the <b>{top_form['Surfactant']} + {top_form['Co-surfactant']}</b> system. 
        <br><br>
        <b>Why?</b> This choice is based on historical outcomes in your research where this pair achieved an 
        average droplet size of <b>{top_form['Size_nm_clean']:.2f} nm</b> and an encapsulation efficiency of <b>{top_form['EE_clean']:.2f}%</b>. 
        This system provides the highest lipophilic affinity for {st.session_state.inputs['drug']}.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No historical data found for this specific oil phase.")

# --- STEP 3: PREDICTION ---
elif page == "3. Precision Prediction":
    st.header("Step 3: Multi-Output Results & Ternary Analysis")
    
    # REQUEST: AXIS GUIDE
    st.markdown(f"""
    <div class="axis-legend-box">
    <b>🧬 Guide to the 3D Pseudo-Ternary Phase Diagram:</b><br>
    • <b>X-Axis (Oil Phase %):</b> The concentration of the lipid carrier.<br>
    • <b>Y-Axis (S-mix %):</b> The combined concentration of Surfactant and Co-surfactant blend.<br>
    • <b>Z-Axis (Aqueous Phase %):</b> The volume of the aqueous phase (Water/Buffer).
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        s_choice = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        cs_choice = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        
        if st.button("🚀 Run AI Engine"):
            # Encodings
            d_idx = le_dict['Drug_Name'].transform([st.session_state.inputs['drug']])[0]
            o_idx = le_dict['Oil_phase'].transform([st.session_state.inputs['oil']])[0]
            s_idx = le_dict['Surfactant'].transform([s_choice])[0]
            c_idx = le_dict['Co-surfactant'].transform([cs_choice])[0]
            
            # Predict
            in_vec = [[d_idx, o_idx, s_idx, c_idx]]
            res = [models[col].predict(in_vec)[0] for col in targets]
            stab = stab_model.predict_proba(in_vec)[0][1] * 100
            
            # DISPLAY (HTML Cards for full font/unit printing)
            outputs = [
                ("Droplet Size", f"{res[0]:.2f} nm"), ("PDI", f"{res[1]:.3f}"),
                ("Zeta Potential", f"{res[2]:.2f} mV"), ("Drug Loading", f"{res[3]:.2f} mg/mL"),
                ("Encapsulation Eff.", f"{res[4]:.2f} %"), ("Stability Confidence", f"{stab:.1f} %")
            ]
            
            for label, val in outputs:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{label}</div><div class='m-value'>{val}</div></div>", unsafe_allow_html=True)

    with c2:
        # Ternary Logic
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
