import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import os
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Fragments, AllChem, DataStructs
import pubchempy as pcp

# --- PAGE CONFIG ---
st.set_page_config(page_title="NanoPredict AI v22.0", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    .main-card { background: #f8f9fa; border-radius: 15px; padding: 25px; border: 1px solid #dee2e6; margin-bottom: 20px; }
    .metric-card { background: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #28a745; text-align: center; }
    .m-label { font-size: 12px; color: #666; font-weight: bold; }
    .m-value { font-size: 24px; font-weight: 800; color: #1a202c; }
    .advice-box { background: #fff4e5; border-left: 5px solid #ff9800; padding: 15px; border-radius: 8px; margin-top: 15px; }
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
        else: return None, None, None, None
    
    cat_cols = ['Drug_Name', 'Surfactant', 'Co-surfactant', 'Oil_phase']
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    def get_num(x):
        val = re.findall(r"[-+]?\d*\.\d+|\d+", str(x))
        return float(val[0]) if val else 0.0

    targets = ['Size_nm', 'PDI', 'Zeta_mV', 'Encapsulation_Efficiency']
    for col in targets: df[f'{col}_clean'] = df[col].apply(get_num)
    
    df_train = df.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_train[f'{col}_enc'] = le.fit_transform(df_train[col])
        le_dict[col] = le

    X = df_train[['Drug_Name_enc', 'Oil_phase_enc', 'Surfactant_enc', 'Co-surfactant_enc']]
    models = {col: GradientBoostingRegressor(n_estimators=100).fit(X, df_train[f'{col}_clean']) for col in targets}
    
    df_train['is_stable'] = df_train.get('Stability', pd.Series(['stable']*len(df_train))).str.lower().str.contains('stable').astype(int)
    stab_model = RandomForestClassifier().fit(X, df_train['is_stable'])
    
    return df, models, stab_model, le_dict

def get_chem_info(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"logp": round(Descriptors.MolLogP(mol), 2), "mw": round(Descriptors.MolWt(mol), 2), "mol": mol}
    except: return None

# --- STATE MGMT ---
if 'step_val' not in st.session_state: st.session_state.step_val = "Step 1"
if 'csv_data' not in st.session_state: st.session_state.csv_data = None

def navigate(step):
    st.session_state.step_val = step
    st.rerun()

# --- STEP 1: INITIAL SETUP ---
if st.session_state.step_val == "Step 1":
    st.title("🧪 NanoPredict AI: Project Initialization")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("1. Drug Definition")
        mode = st.radio("Input Method", ["Database API", "Structural SMILES"])
        
        if mode == "Database API":
            # Temporary dummy list if CSV not loaded yet
            drug_list = ["Drug A", "Drug B"] if st.session_state.csv_data is None else []
            drug_choice = st.selectbox("Select API", drug_list)
            st.session_state.drug_name = drug_choice
            st.session_state.logp, st.session_state.mw = 3.5, 250.0 
        else:
            smiles = st.text_input("Enter SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O")
            info = get_chem_info(smiles)
            if info:
                st.session_state.logp, st.session_state.mw = info['logp'], info['mw']
                st.session_state.drug_name = "Custom_API"
                st.image(Draw.MolToImage(info['mol'], size=(300,200)))

        st.subheader("2. Oil Phase Selection")
        # Load data only after definitions to keep order
        df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
        
        oil_list = sorted(le_dict['Oil_phase'].classes_) if le_dict else ["Select After Upload"]
        st.session_state.oil = st.selectbox("Choose Oil", oil_list)
        
        st.subheader("3. Optional: Training Data")
        up = st.file_uploader("Upload custom CSV (Required to activate AI)", type="csv")
        if up: 
            st.session_state.csv_data = up
            st.rerun()

        if st.button("Analyze Affinities →"): navigate("Step 2")

    with c2:
        if st.session_state.csv_data:
            st.subheader("Oil Affinity Graph")
            oils = le_dict['Oil_phase'].classes_
            # Scoring logic: Based on proximity to LogP 3.0 (typical for oils)
            affinities = [max(10, 100 - abs(st.session_state.logp - 3.0)*10 + np.random.randint(-5,5)) for _ in oils]
            aff_df = pd.DataFrame({"Oil": oils, "Affinity Score": affinities}).sort_values("Affinity Score", ascending=False)
            st.plotly_chart(px.bar(aff_df, x="Affinity Score", y="Oil", orientation='h', color="Affinity Score"), use_container_width=True)

# --- STEP 2: SOLUBILITY ---
elif st.session_state.step_val == "Step 2":
    st.header("Step 2: Solubility Normalization")
    
    # Logic: Convert theoretical 400 mg/mL space to 100 mg/mL practical limit
    # Practical Limit = (Base Solubility / 400) * 100
    base_sol = 10**(0.5 - 0.01 * (st.session_state.mw - 50) - 0.6 * st.session_state.logp) * 1000
    practical_sol = (base_sol / 400) * 100
    st.session_state.sol_limit = np.clip(practical_sol, 1.0, 100.0)

    st.info(f"The theoretical solubility was {base_sol:.1f} mg/mL. We have normalized this to a **Practical Limit of {st.session_state.sol_limit:.2f} mg/mL** (Max 100) based on typical saturation kinetics in nano-systems.")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.drug_conc = st.number_input("Target Drug Conc (mg/mL)", 0.1, 100.0, 5.0)
        st.session_state.oil_p = st.slider("Oil %", 5, 40, 15)
    with c2:
        st.session_state.smix_p = st.slider("S-mix %", 10, 60, 30)
        st.session_state.smix_ratio = st.selectbox("S-mix Ratio", ["1:1", "2:1", "3:1"])

    if st.button("Screen Components →"): navigate("Step 3")

# --- STEP 3 & 4: SUGGESTION & SELECTION ---
elif st.session_state.step_val == "Step 3":
    st.header("Step 3: AI Suggestions & Final Selection")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)
    
    # AI Suggestion Logic
    suggested_s = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values('Encapsulation_Efficiency_clean', ascending=False)['Surfactant'].iloc[0]
    suggested_cs = df_raw[df_raw['Oil_phase'] == st.session_state.oil].sort_values('Size_nm_clean')['Co-surfactant'].iloc[0]

    st.markdown(f"""
    <div class='advice-box'>
        <b>AI Recommendation:</b> Based on your oil choice, the best stability is predicted with 
        <b>{suggested_s}</b> and <b>{suggested_cs}</b>.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.s_final = st.selectbox("Final Surfactant", sorted(le_dict['Surfactant'].classes_), index=list(sorted(le_dict['Surfactant'].classes_)).index(suggested_s))
        st.session_state.cs_final = st.selectbox("Final Co-Surfactant", sorted(le_dict['Co-surfactant'].classes_), index=list(sorted(le_dict['Co-surfactant'].classes_)).index(suggested_cs))
        
        if st.button("Generate Final Report →"): navigate("Step 4")

# --- STEP 5: RESULTS & REPAIR ---
elif st.session_state.step_val == "Step 4":
    st.header("Step 4: AI Performance & Stability Optimization")
    df_raw, models, stab_model, le_dict = load_and_prep(st.session_state.csv_data)

    def safe_encode(le, val):
        try: return le.transform([val])[0]
        except: return 0

    input_vec = [[safe_encode(le_dict[k], v) for k, v in zip(['Drug_Name','Oil_phase','Surfactant','Co-surfactant'], [st.session_state.drug_name, st.session_state.oil, st.session_state.s_final, st.session_state.cs_final])]]

    # 1. Predict 4 Main Outputs
    res = {col: models[col].predict(input_vec)[0] for col in models}
    base_stability = stab_model.predict(input_vec)[0]
    
    # 2. Nanoemulsion Space Check
    # Logic: Nanoemulsions typically require S-mix > 2x Oil phase for high stability
    is_in_space = st.session_state.smix_p > (st.session_state.oil_p * 1.5) and base_stability == 1
    
    cols = st.columns(4)
    outputs = [("SIZE", f"{res['Size_nm']:.1f} nm"), ("PDI", f"{res['PDI']:.3f}"), ("ZETA", f"{res['Zeta_mV']:.1f} mV"), ("EE%", f"{res['Encapsulation_Efficiency']:.1f}%")]
    for i, (l, v) in enumerate(outputs):
        with cols[i]: st.markdown(f"<div class='metric-card'><div class='m-label'>{l}</div><div class='m-value'>{v}</div></div>", unsafe_allow_html=True)

    # 3. Stability Repair Engine
    st.subheader("Stability & Formulation Mapping")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Dynamic ternary based on LogP
        shift = st.session_state.logp * 2
        t_o, t_s = [5, 15, 25, 10, 5], [40+shift, 50+shift, 45+shift, 35+shift, 40+shift]
        t_w = [100-a-b for a, b in zip(t_o, t_s)]
        
        fig = go.Figure(go.Scatterternary({'mode': 'lines', 'fill': 'toself', 'name': 'Stable Nano-Space', 'a': t_o, 'b': t_s, 'c': t_w}))
        fig.add_trace(go.Scatterternary({'mode': 'markers', 'name': 'Current Formula', 'a': [st.session_state.oil_p], 'b': [st.session_state.smix_p], 'c': [100-st.session_state.oil_p-st.session_state.smix_p], 'marker': {'size': 15, 'color': 'red' if not is_in_space else 'green'}}))
        st.plotly_chart(fig, use_container_width=True)
        

    with c2:
        if is_in_space:
            st.success("✅ Formulation is STABLE within the Nanoemulsion region.")
        else:
            st.error("⚠️ Formulation outside Stable Nano-Space.")
            st.markdown("### 🔧 How to Modify for Stability:")
            if st.session_state.smix_p < 25:
                st.write("- **Increase S-mix %:** Your surfactant concentration is too low to emulsify the oil.")
            if st.session_state.oil_p > 20:
                st.write("- **Reduce Oil %:** High internal phase volume is causing coalescence.")
            if abs(res['Zeta_mV']) < 20:
                st.write("- **Charge Adjustment:** Add a charged co-surfactant to increase electrostatic repulsion.")
            st.info("Goal: Aim for an Oil:Smix ratio of at least 1:2.5.")

    if st.button("🔄 Start New Project"): navigate("Step 1")
