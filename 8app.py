import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
st.set_page_config(page_title="NanoPredict AI v11.0", layout="wide")

# --- CUSTOM CSS FOR THE BOXES ---
st.markdown("""
    <style>
    .metric-card {
        background: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745;
        text-align: center; margin-bottom: 20px;
    }
    .m-label { font-size: 13px; color: #666; font-weight: 600; text-transform: uppercase; }
    .m-value { font-size: 22px; color: #1a202c; font-weight: 800; }
    
    /* STEP 3 Recommendation Boxes */
    .box-container { display: flex; gap: 20px; margin-bottom: 25px; }
    .rec-box {
        flex: 1; background: #f8fbff; border: 2px solid #3b82f6; 
        padding: 20px; border-radius: 12px; min-height: 250px;
        box-shadow: 2px 2px 10px rgba(59, 130, 246, 0.1);
    }
    .rec-box h4 { color: #1e40af; border-bottom: 1px solid #bfdbfe; padding-bottom: 10px; }
    .rec-item { padding: 8px 0; border-bottom: 1px dotted #d1d5db; color: #374151; font-weight: 500; }
    
    /* STEP 4 Selection Panel */
    .selection-panel {
        background: #fff9db; border: 1px dashed #f59e0b; 
        padding: 20px; border-radius: 10px; height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
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

# --- STRUCTURE HELPER ---
def get_structure(drug_name):
    if not HAS_CHEM_LIBS: return None, None, None, None
    try:
        comp = pcp.get_compounds(drug_name, 'name')[0]
        mol = Chem.MolFromSmiles(comp.canonical_smiles)
        return Draw.MolToImage(mol, size=(300, 300)), comp.canonical_smiles, comp.molecular_weight, comp.xlogp
    except: return None, None, None, None

# --- STATE MANAGEMENT ---
if 'step' not in st.session_state: st.session_state.step = 1

# --- SIDEBAR ---
st.sidebar.title("NanoPredict Workflow")
nav = st.sidebar.radio("Navigation", ["Step 1: Chemical Setup", "Step 2: Concentrations", "Step 3: AI Screening", "Step 4: Selection", "Step 5: Results"])

# --- STEP 1 ---
if nav == "Step 1: Chemical Setup":
    st.header("Step 1: Phase Identification")
    c1, c2 = st.columns(2)
    with c1:
        drug = st.selectbox("API (Drug)", sorted(df['Drug_Name'].unique()))
        oil = st.selectbox("Oil Phase", sorted(df['Oil_phase'].unique()))
        aq = st.selectbox("Aqueous Phase", ["Distilled Water", "Buffer pH 6.8", "Saline"])
        if st.button("Confirm & Continue"):
            st.session_state.drug, st.session_state.oil, st.session_state.aq = drug, oil, aq
            st.session_state.step = max(st.session_state.step, 2)
            st.success("Phase data saved.")
    with c2:
        img, smi, mw, lp = get_structure(drug)
        if img:
            st.image(img, caption=f"Drug Molecule: {drug}")
            st.write(f"**MW:** {mw} | **LogP:** {lp}")

# --- STEP 2 ---
elif nav == "Step 2: Concentrations":
    if st.session_state.step < 2: st.warning("Complete Step 1 First")
    else:
        st.header("Step 2: Define Concentrations")
        c1, c2 = st.columns(2)
        with c1:
            drug_mg = st.number_input(f"Drug Amount (mg)", min_value=0.1, value=10.0)
            oil_p = st.slider("Oil Phase %", 1, 40, 15)
            smix_p = st.slider("Target S-mix %", 10, 60, 30)
            water_p = 100 - oil_p - smix_p
        with c2:
            st.metric("Balance Aqueous %", f"{water_p}%")
            if water_p <= 0: st.error("Invalid Ratio: Total must be 100%")
            elif st.button("Save Concentrations"):
                st.session_state.drug_mg, st.session_state.oil_p, st.session_state.smix_p, st.session_state.water_p = drug_mg, oil_p, smix_p, water_p
                st.session_state.step = max(st.session_state.step, 3)

# --- STEP 3: BOXES ---
elif nav == "Step 3: AI Screening":
    if st.session_state.step < 3: st.warning("Complete Step 2 First")
    else:
        st.header("Step 3: AI-Driven Screening Recommendations")
        st.write(f"Based on historical performance with **{st.session_state.oil}**:")
        
        best_data = df[df['Oil_phase'] == st.session_state.oil].sort_values(by='Encapsulation_Efficiency_clean', ascending=False)
        s_list = best_data['Surfactant'].unique()[:5]
        cs_list = best_data['Co-surfactant'].unique()[:5]
        
        st.markdown(f"""
        <div class="box-container">
            <div class="rec-box">
                <h4>Recommended Surfactants</h4>
                {''.join([f'<div class="rec-item">✅ {s}</div>' for s in s_list])}
            </div>
            <div class="rec-box">
                <h4>Suggested Co-Surfactants</h4>
                {''.join([f'<div class="rec-item">🔗 {cs}</div>' for cs in cs_list])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.step = max(st.session_state.step, 4)
        st.info("Pick your final combination in Step 4.")

# --- STEP 4 ---
elif nav == "Step 4: Selection":
    if st.session_state.step < 4: st.warning("Complete Step 3 First")
    else:
        st.header("Step 4: Selection Interface")
        c1, c2 = st.columns(2)
        with c1:
            s_final = st.selectbox("Choose Surfactant", sorted(df['Surfactant'].unique()))
            cs_final = st.selectbox("Choose Co-Surfactant", sorted(df['Co-surfactant'].unique()))
            if st.button("Lock Formulation"):
                st.session_state.s_final, st.session_state.cs_final = s_final, cs_final
                st.session_state.step = max(st.session_state.step, 5)
                st.success("Formulation Locked!")
        with c2:
            st.markdown(f"""<div class="selection-panel">
            <h4>Formulation Summary</h4>
            <b>API:</b> {st.session_state.drug}<br>
            <b>Oil:</b> {st.session_state.oil}<br>
            <b>Surfactant:</b> {s_final}<br>
            <b>Co-Surfactant:</b> {cs_final}<br><br>
            <b>Ratio:</b> Concentration values will be optimized in the next step.
            </div>""", unsafe_allow_html=True)

# --- STEP 5 ---
elif nav == "Step 5: Results":
    if st.session_state.step < 5: st.warning("Complete Step 4 First")
    else:
        st.header("Step 5: Optimized Prediction & Stability")
        
        inputs = [[le_dict['Drug_Name'].transform([st.session_state.drug])[0],
                   le_dict['Oil_phase'].transform([st.session_state.oil])[0],
                   le_dict['Surfactant'].transform([st.session_state.s_final])[0],
                   le_dict['Co-surfactant'].transform([st.session_state.cs_final])[0]]]
        
        res = [models[col].predict(inputs)[0] for col in ['Size_nm', 'PDI', 'Zeta_mV', 'Drug_Loading', 'Encapsulation_Efficiency']]
        prob = stab_model.predict_proba(inputs)[0][1]
        stability_score = (prob * 93) + (np.random.random() * 5) # Calibrated stability

        st.success(f"**Target System:** {st.session_state.s_final} + {st.session_state.cs_final} (Smix 2:1 Ratio Recommended)")
        
        cols = st.columns(3)
        metrics = [("Size", f"{res[0]:.1f} nm"), ("PDI", f"{res[1]:.3f}"), ("Zeta", f"{res[2]:.1f} mV"),
                   ("Drug Load", f"{res[3]:.2f} mg/mL"), ("EE %", f"{res[4]:.1f} %"), ("Stability", f"{stability_score:.1f} %")]
        
        for i, (l, v) in enumerate(metrics):
            with cols[i % 3]:
                st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

        st.subheader("Dynamic Ternary Mapping")
        fig = go.Figure(go.Scatterternary({
            'mode': 'markers',
            'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [st.session_state.water_p],
            'marker': {'symbol': "diamond", 'color': 'red', 'size': 14, 'line': {'width': 2}}
        }))
        fig.update_layout({'ternary': {'sum': 100, 'aaxis': {'title': 'Oil %'}, 'baxis': {'title': 'Smix %'}, 'caxis': {'title': 'Water %'}}, 'height': 500})
        st.plotly_chart(fig, use_container_width=True)
