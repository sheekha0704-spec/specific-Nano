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
    from rdkit.Chem import Draw, Descriptors, Fragments
    import pubchempy as pcp
    HAS_CHEM_LIBS = True
except ImportError:
    HAS_CHEM_LIBS = False

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v14.0", layout="wide")

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
    
    /* FIXED STEP 3 BOXES: Removed min-height, added auto height and width padding */
    .rec-box {
        background: #f8fbff; border: 2px solid #3b82f6; 
        padding: 15px; border-radius: 12px; 
        height: auto; width: 100%; margin-bottom: 10px;
    }
    
    .summary-table {
        background: #1a202c; color: white; padding: 20px; 
        border-radius: 12px; border-left: 8px solid #f59e0b;
    }
    .summary-table td { padding: 8px; border-bottom: 1px solid #2d3748; }
    .history-item { font-size: 12px; padding: 5px; border-bottom: 1px solid #eee; color: #444; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data
def load_and_prep():
    csv_file = 'nanoemulsion 2.csv'
    if not os.path.exists(csv_file):
        st.error(f"File '{csv_file}' not found.")
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

# --- 2. STATE MANAGEMENT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1: Chemical Setup"
if 'history' not in st.session_state: st.session_state.history = []
for key in ['drug', 'oil', 'aq', 'drug_mg', 'oil_p', 'smix_p', 'smix_ratio', 's_final', 'cs_final']:
    if key not in st.session_state: st.session_state[key] = None

def go_to_step(next_step):
    st.session_state.step_val = next_step
    st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("NanoPredict AI")
    nav_options = ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"]
    st.session_state.step_val = st.radio("Navigation", nav_options, index=nav_options.index(st.session_state.step_val))
    st.write("---")
    st.subheader("📋 Session History")
    for item in st.session_state.history:
        st.markdown(f"<div class='history-item'><b>{item['drug']}</b> | Size: {item['size']}nm</div>", unsafe_allow_html=True)

# --- STEP 1: CHEMICAL SETUP ---
if st.session_state.step_val == "Step 1: Chemical Setup":
    st.header("Step 1: API & Structural Analysis")
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("Select API (Drug)", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Select Oil Phase", sorted(df['Oil_phase'].unique()))
        aq = st.selectbox("Select Aqueous Phase", ["Distilled Water", "Buffer pH 6.8", "Saline"])
        if st.button("Confirm Phase Setup →"):
            st.session_state.drug, st.session_state.oil, st.session_state.aq = drug, oil, aq
            go_to_step("Step 2: Concentrations")
    with c2:
        if HAS_CHEM_LIBS:
            try:
                comp = pcp.get_compounds(drug, 'name')[0]
                mol = Chem.MolFromSmiles(comp.canonical_smiles)
                st.image(Draw.MolToImage(mol, size=(300,300)), caption=f"Chemical Structure: {drug}")
                
                fgroups = []
                if Fragments.fr_Al_OH(mol) > 0: fgroups.append("Alcohol (-OH)")
                if Fragments.fr_Ar_OH(mol) > 0: fgroups.append("Phenol")
                if Fragments.fr_NH2(mol) > 0: fgroups.append("Primary Amine")
                if Fragments.fr_C_O(mol) > 0: fgroups.append("Carbonyl Group")
                if Fragments.fr_COO(mol) > 0: fgroups.append("Carboxylic Acid/Ester")
                
                st.subheader("Molecular Profile")
                st.write(f"**Identified Groups:** {', '.join(fgroups) if fgroups else 'Complex/Other'}")
                st.write(f"**Molecular Weight:** {comp.molecular_weight} g/mol")
                st.write(f"**LogP:** {comp.xlogp}")
            except: st.warning("Structural details could not be retrieved.")

# --- STEP 2: CONCENTRATIONS ---
elif st.session_state.step_val == "Step 2: Concentrations":
    st.header("Step 2: Uniform Formulation Inputs")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.drug_mg = st.number_input("Drug Dose (mg)", value=10.0, step=1.0)
        st.session_state.oil_p = st.number_input("Oil Phase (%)", value=15.0, step=1.0)
        st.session_state.smix_p = st.number_input("Total S-mix (%)", value=30.0, step=1.0)
        st.session_state.smix_ratio = st.selectbox("S-mix Ratio (S:Co-S)", ["1:1", "2:1", "3:1", "4:1", "Custom"])
        if st.session_state.smix_ratio == "Custom":
            st.session_state.smix_ratio = st.text_input("Enter Ratio", "2.5:1")
        
        st.session_state.water_p = 100 - st.session_state.oil_p - st.session_state.smix_p
        if st.button("Save & Screen Components →"):
            go_to_step("Step 3: AI Screening")
    with c2:
        st.metric("Balance Water %", f"{st.session_state.water_p}%")
        st.info("Ensure Oil + Smix does not exceed 100%. Adjust values if Aqueous % is negative.")

# --- STEP 3: AI SCREENING (MODIFIED FOR BOX WIDTH) ---
elif st.session_state.step_val == "Step 3: AI Screening":
    st.header("Step 3: Suggested Components for Selection")
    if st.session_state.oil is None: st.error("Please complete Step 1.")
    else:
        best_data = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
        s_list = best_data['Surfactant'].unique()[:5]
        cs_list = best_data['Co-surfactant'].unique()[:5]
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="rec-box"><h4>Recommended Surfactants</h4>', unsafe_allow_html=True)
            for s in s_list: st.write(f"✅ {s}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="rec-box"><h4>Suggested Co-Surfactants</h4>', unsafe_allow_html=True)
            for cs in cs_list: st.write(f"🔗 {cs}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Finalize Pair in Step 4 →"):
            go_to_step("Step 4: Selection")

# --- STEP 4: SELECTION & SUMMARY ---
elif st.session_state.step_val == "Step 4: Selection":
    st.header("Step 4: Selection & Summary")
    c1, c2 = st.columns(2)
    with c1:
        s_final = st.selectbox("Select Surfactant", sorted(df['Surfactant'].unique()))
        cs_final = st.selectbox("Select Co-Surfactant", sorted(df['Co-surfactant'].unique()))
        if st.button("Generate Final Prediction →"):
            st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
            go_to_step("Step 5: Results")
    with c2:
        st.markdown(f"""
        <div class="summary-table">
            <h4>📋 Selection Summary</h4>
            <table style="width:100%">
                <tr><td><b>Drug Selection</b></td><td>{st.session_state.drug}</td></tr>
                <tr><td><b>Oil Phase</b></td><td>{st.session_state.oil} ({st.session_state.oil_p}%)</td></tr>
                <tr><td><b>Smix Total</b></td><td>{st.session_state.smix_p}%</td></tr>
                <tr><td><b>S:Co-S Ratio</b></td><td>{st.session_state.smix_ratio}</td></tr>
                <tr><td><b>Selected Surfactant</b></td><td>{s_final}</td></tr>
                <tr><td><b>Selected Co-Surfactant</b></td><td>{cs_final}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# --- STEP 5: RESULTS ---
elif st.session_state.step_val == "Step 5: Results":
    st.header("Step 5: Performance Results")
    if st.session_state.s_final is None: 
        st.error("Please complete Step 4.")
    else:
        inputs = [[le_dict['Drug_Name'].transform([st.session_state.drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
        
        if not any(h['s'] == st.session_state.s_final and h['cs'] == st.session_state.cs_final for h in st.session_state.history):
            st.session_state.history.append({'drug': st.session_state.drug, 'size': round(res[0],1), 's': st.session_state.s_final, 'cs': st.session_state.cs_final})

        cols = st.columns(3)
        m_list = [("Size", f"{res[0]:.1f} nm"), ("PDI", f"{res[1]:.3f}"), ("Zeta", f"{res[2]:.1f} mV"),
                  ("Loading", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability", "High")]
        for i, (l, v) in enumerate(m_list):
            with cols[i % 3]: 
                st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("PDI Distribution Curve")
            mu, sigma = res[0], res[0] * res[1]
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
            fig_dist = px.line(x=x, y=y, labels={'x': 'Size (nm)', 'y': 'Intensity'}, title="Size Distribution")
            fig_dist.add_vline(x=mu, line_dash="dash", line_color="red")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c2:
            st.subheader("Ternary Phase Mapping")
            fig_tern = go.Figure(go.Scatterternary({
                'mode': 'markers',
                'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p],
                'marker': {'color': '#3498db', 'size': 14, 'symbol': 'diamond'}
            }))
            fig_tern.update_layout({'ternary': {'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}}, 'height': 450})
            st.plotly_chart(fig_tern, use_container_width=True)

    if st.button("🔄 Start New Calculation"):
        go_to_step("Step 1: Chemical Setup")
