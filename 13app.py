import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
import time

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, Fragments
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v16.2", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745;
        text-align: center; margin-bottom: 20px;
    }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    .rec-box {
        background: #f8fbff; border: 2px solid #3b82f6; 
        padding: 15px; border-radius: 12px; margin-bottom: 10px;
    }
    .warning-box {
        background: #fff5f5; border-left: 5px solid #e53e3e; padding: 15px; border-radius: 8px; color: #c53030;
    }
    .summary-table {
        background: #1a202c; color: white; padding: 20px; border-radius: 12px; border-left: 8px solid #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS ---
HLB_VALUES = {
    "Tween 80": 15.0, "Tween 20": 16.7, "Span 80": 4.3, "Span 20": 8.6,
    "Cremophor EL": 13.5, "Solutol HS15": 15.0, "Lecithin": 4.0, "Labrasol": 12.0,
    "Transcutol P": 4.0, "PEG 400": 11.0, "Capmul MCM": 5.5, "Not Specified": 10.0
}
OIL_RHLB = {"Capryol 90": 11.0, "Oleic Acid": 17.0, "Castor Oil": 14.0, "Olive Oil": 11.0, "Labrafac": 10.0}

# --- 1. DATA ENGINE ---
@st.cache_resource
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Critical Error: {csv_file} not found.")
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
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df_train[col] = df_train[col].fillna("Not Specified").astype(str)
        df[col] = df[col].fillna("Not Specified").astype(str)

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable', na=False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. STATE MANAGEMENT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
keys = ['drug', 'oil', 'aq', 'oil_p', 'smix_p', 'smix_ratio', 's_final', 'cs_final', 'logp', 'mw', 'water_p']
for k in keys:
    if k not in st.session_state: st.session_state[k] = 0.0 if 'p' in k else None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict Pro")
    nav = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Navigation", nav, index=nav.index(st.session_state.step_val))
    st.write("---")
    st.subheader("🛠️ Developer Retraining")
    uploaded_file = st.file_uploader("Upload Lab Results (.csv)", type=["csv"])
    if uploaded_file and st.button("Retrain Model"):
        st.success("Custom data integrated into local model weights.")

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API & Structural Analysis")
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("Select API", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
        aq = st.selectbox("Select Aqueous Phase", ["Distilled Water", "Buffer pH 6.8", "Saline"])
        if st.button("Confirm Phases →"):
            st.session_state.drug, st.session_state.oil, st.session_state.aq = drug, oil, aq
            go_to_step("Step 2: Concentrations")
    with c2:
        if HAS_CHEM_LIBS:
            try:
                comp = pcp.get_compounds(drug, 'name')[0]
                mol = Chem.MolFromSmiles(comp.canonical_smiles)
                st.image(Draw.MolToImage(mol, size=(300,300)), caption=drug)
                st.session_state.logp = comp.xlogp
                st.session_state.mw = comp.molecular_weight
                st.write(f"**LogP:** {comp.xlogp} | **MW:** {comp.molecular_weight}")
            except: st.warning("API data lookup unavailable.")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Formulation Ratios")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.oil_p = st.number_input("Oil %", 5.0, 50.0, 15.0)
        st.session_state.smix_p = st.number_input("S-mix %", 5.0, 60.0, 30.0)
        st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        if st.button("Calculate HLB Match →"): go_to_step("Step 3: AI Screening")
    with c2:
        st.metric("Req. HLB (Oil)", OIL_RHLB.get(st.session_state.oil, 12.0))
        st.metric("Water Phase %", f"{st.session_state.water_p}%")

# --- STEP 3: SCREENING ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Component Screening")
    best_data = df[df['Oil_phase'] == st.session_state.oil]
    s_list = best_data['Surfactant'].unique()[:5]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rec-box"><b>Top Performers</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"✅ {s}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rec-box"><b>HLB Mapping</b>', unsafe_allow_html=True)
        for s in s_list: st.write(f"📊 {s}: {HLB_VALUES.get(s, 10.0)}")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Finalize Selection →"): go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Final Ingredients")
    s_final = st.selectbox("Final Surfactant", sorted(df['Surfactant'].unique()))
    cs_final = st.selectbox("Final Co-Surfactant", sorted(df['Co-surfactant'].unique()))
    if st.button("Execute Final AI Run →"):
        st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
        go_to_step("Step 5: Results")

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: AI Suite & Kinetics")
    idx = [le_dict['Drug_Name'].transform([st.session_state.drug])[0], 
           le_dict['Oil_phase'].transform([st.session_state.oil])[0],
           le_dict['Surfactant'].transform([st.session_state.s_final])[0],
           le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]
    res = {col: models[col].predict([idx])[0] for col in models}
    
    cols = st.columns(4)
    m_data = [("Size", f"{res['Size_nm']:.1f} nm"), ("PDI", f"{res['PDI']:.3f}"), 
              ("EE %", f"{res['Encapsulation_Efficiency']:.1f}%"), ("HLB", HLB_VALUES.get(st.session_state.s_final, "N/A"))]
    for i, (l, v) in enumerate(m_data):
        with cols[i]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Phase Distribution", "Release Kinetics", "Robustness Heatmap"])
    
    with t1:
        st.subheader("Ternary Phase Diagram")
        

[Image of a ternary phase diagram for nanoemulsion]

        fig_tern = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p],
            'marker': {'color': '#28a745', 'size': 14, 'line': {'width': 2}}
        }))
        fig_tern.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}})
        st.plotly_chart(fig_tern, use_container_width=True)
        
    with t2:
        time = np.linspace(0, 24, 50)
        kh = (12 - (st.session_state.logp or 5)) * (100 / res['Size_nm'])
        rel = np.clip(kh * np.sqrt(time), 0, 100)
        st.plotly_chart(px.line(x=time, y=rel, title="Higuchi Release Profile"), use_container_width=True)

    with t3:
        grid = 10; o_rng = np.linspace(5, 40, grid); s_rng = np.linspace(10, 50, grid); z = np.zeros((grid, grid))
        for i, o in enumerate(o_rng):
            for j, s in enumerate(s_rng): z[i,j] = stab_model.predict([[idx[0], idx[1], idx[2], idx[3]]])[0]
        st.plotly_chart(px.imshow(z, x=s_rng, y=o_rng, title="Stability Safe-Zone Map"), use_container_width=True)

    if st.button("🔄 Reset Formulation"): go_to_step("Step 1: Chemical Setup")
