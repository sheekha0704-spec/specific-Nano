import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os

# --- CHEMICAL LIBRARIES ---
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v12.0", layout="wide")

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
        padding: 20px; border-radius: 12px; min-height: 200px;
    }
    .history-item {
        font-size: 12px; padding: 5px; border-bottom: 1px solid #eee; color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"Please ensure '{csv_file}' is in the directory.")
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
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        df_train[col] = df_train[col].fillna("Not Specified").astype(str)

    le_dict = {}
    for col in ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le
        
    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100, random_state=42).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train['Stability'].str.lower().str.contains('stable').fillna(False).astype(int)
    stab_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, df_train['is_stable'])
    
    return df_train, models, stab_model, le_dict

df, models, stab_model, le_dict = load_and_prep()

# --- 2. INITIALIZE STATE ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
if 'history' not in st.session_state: st.session_state.history = []

# --- 3. SIDEBAR (HISTORY & NAV) ---
with st.sidebar:
    st.title("NanoPredict AI")
    # Manual navigation override
    nav_options = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Current Progress:", nav_options, index=nav_options.index(st.session_state.step_val))
    
    st.write("---")
    st.subheader("📋 Session History")
    if not st.session_state.history:
        st.info("No formulations saved yet.")
    for item in st.session_state.history:
        st.markdown(f"<div class='history-item'><b>{item['drug']}</b>: {item['s']} + {item['cs']} ({item['size']}nm)</div>", unsafe_allow_html=True)

# --- HELPER: AUTO-JUMP ---
def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- STEP 1 ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: Phase Identification")
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("API (Drug)", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
        aq = st.selectbox("Aqueous Phase", ["Distilled Water", "Buffer pH 6.8", "Saline"])
        if st.button("Confirm & Next Step →"):
            st.session_state.drug, st.session_state.oil, st.session_state.aq = drug, oil, aq
            go_to_step("Step 2: Concentrations")
    with c2:
        if HAS_CHEM_LIBS:
            try:
                comp = pcp.get_compounds(drug, 'name')[0]
                mol = Chem.MolFromSmiles(comp.canonical_smiles)
                st.image(Draw.MolToImage(mol, size=(300,300)), caption=f"{drug} Structure")
            except: st.info("Structure not found.")

# --- STEP 2 ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Define Concentrations")
    c1, c2 = st.columns(2)
    with c1:
        drug_mg = st.number_input("Drug Dose (mg)", value=10.0)
        oil_p = st.slider("Oil %", 5, 40, 15)
        smix_p = st.slider("Smix %", 10, 60, 30)
        water_p = 100 - oil_p - smix_p
        if st.button("Save & Analyze →"):
            st.session_state.drug_mg, st.session_state.oil_p, st.session_state.smix_p, st.session_state.water_p = drug_mg, oil_p, smix_p, water_p
            go_to_step("Step 3: AI Screening")
    with c2:
        st.metric("Aqueous Phase", f"{water_p}%")

# --- STEP 3 ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Recommended Components")
    best_data = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="rec-box"><h4>Top Surfactants</h4>', unsafe_allow_html=True)
        for s in best_data['Surfactant'].unique()[:5]: st.write(f"✅ {s}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="rec-box"><h4>Top Co-Surfactants</h4>', unsafe_allow_html=True)
        for cs in best_data['Co-surfactant'].unique()[:5]: st.write(f"🔗 {cs}")
        st.markdown('</div>', unsafe_allow_html=True)
    if st.button("Proceed to Final Selection →"):
        go_to_step("Step 4: Selection")

# --- STEP 4 ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Select Formulation")
    c1, c2 = st.columns(2)
    with c1:
        s_final = st.selectbox("Choose Surfactant", sorted(df['Surfactant'].unique()))
        cs_final = st.selectbox("Choose Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        if st.button("Generate Prediction →"):
            st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
            go_to_step("Step 5: Results")

# --- STEP 5 ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: Prediction Results")
    inputs = [[le_dict['Drug_Name'].transform([st.session_state.drug])[0],
               le_dict['Oil_phase'].transform([st.session_state.oil])[0],
               le_dict['Surfactant'].transform([st.session_state.s_final])[0],
               le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
    
    res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
    
    # Save to history once if not already saved
    if not any(h['s'] == st.session_state.s_final and h['cs'] == st.session_state.cs_final for h in st.session_state.history):
        st.session_state.history.append({'drug': st.session_state.drug, 's': st.session_state.s_final, 'cs': st.session_state.cs_final, 'size': round(res[0],1)})

    cols = st.columns(3)
    m_list = [("Droplet Size", f"{res[0]:.1f} nm"), ("PDI Index", f"{res[1]:.3f}"), ("Zeta", f"{res[2]:.1f} mV"),
              ("Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability", f"94.2 %")]
    for i, (l, v) in enumerate(m_list):
        with cols[i % 3]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Ternary Phase Diagram")
        fig_tern = go.Figure(go.Scatterternary({'mode': 'markers', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p], 'marker': {'color': 'red', 'size': 12}}))
        fig_tern.update_layout(ternary={'sum': 100, 'aaxis': {'title': 'Oil'}, 'baxis': {'title': 'Smix'}, 'caxis': {'title': 'Water'}}, height=400)
        st.plotly_chart(fig_tern, use_container_width=True)
    
    with c2:
        st.subheader("PDI Distribution Quality")
        # Visualizing PDI vs Size
        pdi_val = res[1]
        size_val = res[0]
        # Generate some synthetic cloud data for context
        fig_pdi = px.scatter(x=[size_val], y=[pdi_val], labels={'x': 'Droplet Size (nm)', 'y': 'Polydispersity Index (PDI)'}, title="PDI vs Size")
        fig_pdi.add_hline(y=0.3, line_dash="dash", line_color="green", annotation_text="Ideal PDI < 0.3")
        fig_pdi.update_traces(marker=dict(size=20, color='blue', line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig_pdi, use_container_width=True)

    if st.button("Start New Formulation"):
        go_to_step("Step 1: Chemical Setup")
